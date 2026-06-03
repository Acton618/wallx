import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_ode_fixed_steps_lerobot_images import set_seed
from scripts.profile_v5_ode_cache_lerobot_images import (
    CASE_SPECS,
    action_error,
    case_runtime_kwargs,
    pct_delta,
)
from scripts.profile_vispruner_lerobot_media import (
    average_timing_dict,
    build_common_fields,
    format_text,
    load_model,
    mean,
    sync_if_cuda,
)
from wall_x.data.utils import preprocesser_call


def make_video_items(args):
    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob(args.video_glob))
    if not videos:
        raise FileNotFoundError(f"No videos matched {video_dir / args.video_glob}")
    if args.num_videos > 0:
        pick = np.linspace(0, max(0, len(videos) - 1), args.num_videos, dtype=int)
        videos = [videos[int(i)] for i in pick]
    return [
        {
            "video_path": video_path,
            "prompt": args.prompt,
            "source": f"video={video_path}",
        }
        for video_path in videos
    ]


def sample_video_frames(video_path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    if frame_count <= 0:
        indices = list(range(num_frames))
    else:
        indices = np.linspace(0, max(0, frame_count - 1), num_frames, dtype=int).tolist()

    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, {"frame_count": frame_count, "fps": fps, "indices": indices}


def make_video_batch(model, item, args):
    frames, meta = sample_video_frames(item["video_path"], args.video_frames)
    # Use the repository's unified V1/V2 video preprocessing path. Some Wall-X
    # checkpoints expose only image_processor; preprocesser_call falls back to it
    # and still returns pixel_values_videos/video_grid_thw.
    inputs = preprocesser_call(
        processor=model.processor,
        text=[format_text(item["prompt"], "video", args.pred_horizon)],
        images=None,
        videos=[frames],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=args.max_length,
    )
    video_count = int(inputs["video_grid_thw"].shape[0])
    inputs["second_per_grid_ts"] = torch.full(
        (video_count,), 1.0 / meta["fps"], dtype=torch.float32
    )
    inputs = build_common_fields(inputs, model.processor, args)
    return inputs.to(args.device), meta


def video_token_count_from_grid(video_grid_thw, spatial_merge_size: int) -> int:
    total = 0
    for grid in video_grid_thw:
        temporal, height, width = [int(x) for x in grid]
        total += temporal * (height // spatial_merge_size) * (width // spatial_merge_size)
    return total


def expected_video_tokens_after_prune(args, model, video_grid_thw) -> int:
    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    before = video_token_count_from_grid(video_grid_thw, spatial_merge_size)
    # V4 video pruning happens inside generate_flow_action(), after processor
    # expansion and before token embedding. The batch still contains the original
    # video placeholder count, so the profiling script reports the expected
    # internal count from the same keep_ratio/min_tokens used by VisPruner.
    if not (args.enable_pruning and args.prune_video):
        return before

    kept = 0
    for grid in video_grid_thw:
        length = video_token_count_from_grid([grid], spatial_merge_size)
        kept += max(1, int(np.ceil(length * float(args.keep_ratio))))
    return kept


@torch.no_grad()
def run_case(args, model, items, case_name):
    records = []
    timing_counts = {}
    runtime_kwargs = case_runtime_kwargs(args, case_name)

    for idx, item in enumerate(items, start=1):
        sync_if_cuda(args.device)
        prepare_start = time.perf_counter()
        batch, video_meta = make_video_batch(model, item, args)
        sync_if_cuda(args.device)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        video_grid_thw = batch["video_grid_thw"].detach().cpu().tolist()
        tokens_before = video_token_count_from_grid(
            video_grid_thw,
            int(model.config.vision_config.spatial_merge_size),
        )
        tokens_after = expected_video_tokens_after_prune(args, model, video_grid_thw)
        second_per_grid_ts = batch["second_per_grid_ts"].detach().cpu().tolist()

        call_kwargs = dict(
            **batch,
            action_dim=args.action_dim,
            action_horizon=args.pred_horizon,
            mode="predict",
            predict_mode="diffusion",
            unnorm=False,
            num_inference_timesteps=args.num_inference_timesteps,
        )

        for warmup_idx in range(args.warmup):
            seed = args.base_seed + idx * 1000 - args.warmup + warmup_idx
            set_seed(seed, args.device)
            _ = model(
                **call_kwargs,
                profile_timing=False,
                print_timing=False,
                **runtime_kwargs,
            )

        timing_dicts = []
        early_infos = []
        cache_infos = []
        last_action = None
        for iter_idx in range(args.iters):
            seed = args.base_seed + idx * 1000 + iter_idx
            set_seed(seed, args.device)
            out = model(
                **call_kwargs,
                profile_timing=True,
                print_timing=False,
                **runtime_kwargs,
            )
            timing_dicts.append(dict(out.get("timing_results_ms", {})))
            early_infos.append(dict(out.get("ode_early_stop_info", {})))
            cache_infos.append(dict(out.get("ode_cache_info", {})))
            if out.get("predict_action") is not None:
                last_action = out["predict_action"].detach().cpu().to(torch.float32).numpy()

        timing_counts = dict(out.get("timing_counts", timing_counts))
        avg_timing = average_timing_dict(timing_dicts)
        actual_steps = [info.get("actual_steps", args.num_inference_timesteps) for info in early_infos]
        stopped = [bool(info.get("stopped", False)) for info in early_infos]
        last_delta = [info.get("last_delta", None) for info in early_infos]
        last_delta = [float(x) for x in last_delta if x is not None]
        cache_calls = [int(info.get("calls", 0)) for info in cache_infos]
        cache_refreshes = [int(info.get("refreshes", 0)) for info in cache_infos]
        cache_hits = [int(info.get("hits", 0)) for info in cache_infos]
        cache_hit_rates = [float(info.get("hit_rate", 0.0)) for info in cache_infos]

        record = {
            "case": case_name,
            "case_label": CASE_SPECS[case_name]["label"],
            "index": idx,
            "source": item["source"],
            "external_prepare_batch_ms": prepare_ms,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
            "video_meta": video_meta,
            "predict_action_last": last_action.reshape(-1).tolist() if last_action is not None else None,
            "timings_ms": avg_timing,
            "ode_early_stop": {
                "runtime_kwargs": runtime_kwargs,
                "actual_steps_mean": mean(actual_steps),
                "actual_steps_min": min(actual_steps),
                "actual_steps_max": max(actual_steps),
                "postfix_steps_mean": mean(x - 1 for x in actual_steps),
                "stopped_rate": mean(1.0 if x else 0.0 for x in stopped),
                "last_delta_mean": mean(last_delta),
            },
            "ode_cache": {
                "enabled": bool(runtime_kwargs.get("ode_cache_enable", False)),
                "interval": runtime_kwargs.get("ode_cache_interval", None),
                "start_step": runtime_kwargs.get("ode_cache_start_step", None),
                "calls_mean": mean(cache_calls),
                "refreshes_mean": mean(cache_refreshes),
                "hits_mean": mean(cache_hits),
                "hit_rate_mean": mean(cache_hit_rates),
            },
        }
        records.append(record)
        print(
            f"[V5_VIDEO_CACHE] {case_name} {idx}/{len(items)} "
            f"video_tokens={tokens_before}->{tokens_after} grid={video_grid_thw} "
            f"updates={record['ode_early_stop']['actual_steps_mean']:.2f} "
            f"cache_hit={record['ode_cache']['hit_rate_mean']:.2%} "
            f"total_ms={avg_timing.get('total_time', 0.0):.3f} "
            f"ode_ms={avg_timing.get('ode_integration', 0.0):.3f}",
            flush=True,
        )
    return records, timing_counts


def summarize(records):
    timing_keys = sorted({key for rec in records for key in rec["timings_ms"]})
    return {
        "num_samples": len(records),
        "tokens_before": mean(rec["tokens_before"] for rec in records),
        "tokens_after": mean(rec["tokens_after"] for rec in records),
        "external_prepare_batch_ms": mean(rec["external_prepare_batch_ms"] for rec in records),
        "actual_steps_mean": mean(rec["ode_early_stop"]["actual_steps_mean"] for rec in records),
        "actual_steps_min": min(rec["ode_early_stop"]["actual_steps_min"] for rec in records),
        "actual_steps_max": max(rec["ode_early_stop"]["actual_steps_max"] for rec in records),
        "postfix_steps_mean": mean(rec["ode_early_stop"]["postfix_steps_mean"] for rec in records),
        "stopped_rate": mean(rec["ode_early_stop"]["stopped_rate"] for rec in records),
        "last_delta_mean": mean(rec["ode_early_stop"]["last_delta_mean"] for rec in records),
        "cache_refreshes_mean": mean(rec["ode_cache"]["refreshes_mean"] for rec in records),
        "cache_hits_mean": mean(rec["ode_cache"]["hits_mean"] for rec in records),
        "cache_hit_rate_mean": mean(rec["ode_cache"]["hit_rate_mean"] for rec in records),
        "timings_ms": {
            key: mean(rec["timings_ms"].get(key, 0.0) for rec in records)
            for key in timing_keys
        },
    }


def append_stage_table(report, summaries):
    cases = list(CASE_SPECS)
    priority = [
        "total_time",
        "external_prepare_batch_ms",
        "embed_processing",
        "vision_video_forward",
        "scatter_video_embeds",
        "position_encoding",
        "action_initialization",
        "prefetch_forward",
        "prefill_transformer",
        "cache_preprocessing",
        "ode_integration",
        "ode_transformer_total",
        "ode_action_embed_total",
        "ode_prepare_inputs",
        "ode_action_head_total",
        "postprocessing",
    ]
    all_keys = sorted(set().union(*(set(summary["timings_ms"]) for summary in summaries.values())))
    report.append("| stage | " + " | ".join(f"{case}_ms" for case in cases) + " | cache_i2_delta_vs_fixed |")
    report.append("|---|" + "---:|" * (len(cases) + 1))
    seen = set()
    for key in priority + all_keys:
        if key in seen:
            continue
        seen.add(key)
        vals = []
        for case in cases:
            summary = summaries[case]
            if key == "external_prepare_batch_ms":
                vals.append(summary["external_prepare_batch_ms"])
            else:
                vals.append(summary["timings_ms"].get(key, 0.0))
        if all(v == 0.0 for v in vals):
            continue
        delta = pct_delta(vals[1], vals[0]) if len(vals) > 1 else 0.0
        report.append(
            f"| `{key}` | "
            + " | ".join(f"{val:.3f}" for val in vals)
            + f" | {delta:+.2f}% |"
        )
    report.append("")


def write_outputs(args, records, counts):
    summaries = {case: summarize(records[case]) for case in CASE_SPECS}
    errors = {
        case: action_error(records["fixed_10"], records[case])
        for case in CASE_SPECS
        if case != "fixed_10"
    }
    fixed = summaries["fixed_10"]
    fixed_total = fixed["timings_ms"].get("total_time", 0.0)
    fixed_ode = fixed["timings_ms"].get("ode_integration", 0.0)

    report = [
        "# Wall-X V5 ODE Cache Video Dataset Report\n",
        f"- video_dir: `{args.video_dir}`",
        f"- video_glob: `{args.video_glob}`",
        f"- num_videos: `{args.num_videos}`",
        f"- video_frames_per_clip: `{args.video_frames}`",
        f"- prompt: `{args.prompt}`",
        f"- model_path: `{args.model_path}`",
        f"- media_type: `video`",
        f"- vispruner.prune_video: `{args.prune_video}`",
        f"- vispruner.keep_ratio: `{args.keep_ratio}`",
        f"- num_inference_timesteps: `{args.num_inference_timesteps}`",
        f"- warmup: `{args.warmup}`",
        f"- iters: `{args.iters}`",
        f"- base_seed: `{args.base_seed}`",
        f"- device: `{args.device}`",
        "",
        "## V5 Cases\n",
        "| case | label | early_stop | cache | interval | start_step |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in CASE_SPECS:
        kwargs = case_runtime_kwargs(args, case)
        report.append(
            f"| `{case}` | {CASE_SPECS[case]['label']} | "
            f"`{kwargs.get('ode_early_stop_enable')}` | "
            f"`{kwargs.get('ode_cache_enable')}` | "
            f"`{kwargs.get('ode_cache_interval', '-')}` | "
            f"`{kwargs.get('ode_cache_start_step', '-')}` |"
        )
    report.extend([
        "",
        "## Summary\n",
        "| case | video_tokens_before | expected_video_tokens_after | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | cache_refreshes | cache_hits | cache_hit_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in CASE_SPECS:
        summary = summaries[case]
        total = summary["timings_ms"].get("total_time", 0.0)
        ode = summary["timings_ms"].get("ode_integration", 0.0)
        err = errors.get(case, {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0})
        report.append(
            f"| `{case}` | {summary['tokens_before']:.2f} | {summary['tokens_after']:.2f} | "
            f"{total:.3f} | {pct_delta(total, fixed_total):+.2f}% | "
            f"{ode:.3f} | {pct_delta(ode, fixed_ode):+.2f}% | "
            f"{summary['actual_steps_mean']:.2f} | "
            f"{summary['cache_refreshes_mean']:.2f} | {summary['cache_hits_mean']:.2f} | "
            f"{summary['cache_hit_rate_mean']:.2%} | "
            f"{err['mae']:.6f} | {err['rmse']:.6f} | {err['max_abs']:.6f} |"
        )

    report.extend([
        "",
        "## Interpretation\n",
        "- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.",
        "- `expected_video_tokens_after` is the V4 internal video token count after VisPruner. The raw batch still contains the original placeholders before the model prunes them.",
        "- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.",
        "- Accuracy is action difference against `fixed_10` under the same video sample and seed.",
        "- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.",
        "",
        "## Stage Timing\n",
    ])
    append_stage_table(report, summaries)

    report.extend([
        "## Per-Video Paired Results\n",
        "| idx | source | video_grid_thw | second_per_grid_ts | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    case_errors = {case: {row["index"]: row for row in err["rows"]} for case, err in errors.items()}
    for fixed_rec, cache2_rec, cache3_rec, trade_rec in zip(
        records["fixed_10"], records["cache_i2"], records["cache_i3"], records["early_tradeoff"]
    ):
        idx = fixed_rec["index"]
        report.append(
            f"| {idx} | `{fixed_rec['source']}` | `{fixed_rec['video_grid_thw']}` | `{fixed_rec['second_per_grid_ts']}` | "
            f"{fixed_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{cache2_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{cache3_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{trade_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{case_errors['cache_i2'][idx]['action_mae']:.6f} | "
            f"{case_errors['cache_i3'][idx]['action_mae']:.6f} | "
            f"{case_errors['early_tradeoff'][idx]['action_mae']:.6f} |"
        )

    report.extend(["", "## Raw Results", f"- `{args.results_json}`"])

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text("\n".join(report), encoding="utf-8")
    Path(args.results_json).write_text(
        json.dumps(
            {
                "args": vars(args),
                "case_specs": CASE_SPECS,
                "records": records,
                "summaries": summaries,
                "action_errors_vs_fixed_10": errors,
                "timing_counts": counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/wall_x/pretrained/wall-oss-fast")
    parser.add_argument("--video-dir", default="/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg")
    parser.add_argument("--video-glob", default="episode_*.mp4")
    parser.add_argument("--num-videos", type=int, default=50)
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--enable-pruning", action="store_true", default=True)
    parser.add_argument("--disable-pruning", dest="enable_pruning", action="store_false")
    parser.add_argument("--pruned-strategy", default="topk_attention")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--prune-video", action="store_true", default=False)
    parser.add_argument("--no-prune-video", dest="prune_video", action="store_false")
    parser.add_argument("--predictor-checkpoint", default=None)
    parser.add_argument("--predictor-source", default="image_embeds")
    parser.add_argument("--predictor-early-layer", type=int, default=None)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-inference-timesteps", type=int, default=10)
    parser.add_argument("--safe-threshold", type=float, default=0.2)
    parser.add_argument("--safe-min-steps", type=int, default=2)
    parser.add_argument("--safe-patience", type=int, default=1)
    parser.add_argument("--tradeoff-threshold", type=float, default=0.3)
    parser.add_argument("--tradeoff-min-steps", type=int, default=8)
    parser.add_argument("--tradeoff-patience", type=int, default=1)
    parser.add_argument("--cache-interval-2", type=int, default=2)
    parser.add_argument("--cache-interval-3", type=int, default=3)
    parser.add_argument("--cache-start-step", type=int, default=2)
    parser.add_argument(
        "--report-path",
        default="/root/autodl-tmp/wall_x/workspace/v5_ode_cache/libero_50video_v5_ode_cache_report.md",
    )
    parser.add_argument(
        "--results-json",
        default="/root/autodl-tmp/wall_x/workspace/v5_ode_cache/libero_50video_v5_ode_cache_results.json",
    )
    args = parser.parse_args()

    items = make_video_items(args)
    model = load_model(args, args.enable_pruning)

    records = {}
    counts = {}
    for case_name in CASE_SPECS:
        records[case_name], counts[case_name] = run_case(args, model, items, case_name)

    write_outputs(args, records, counts)
    print(f"[V5_VIDEO_CACHE] report={args.report_path}", flush=True)
    print(f"[V5_VIDEO_CACHE] results_json={args.results_json}", flush=True)

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
