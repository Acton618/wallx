import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.profile_vispruner_lerobot_media import (
    average_timing_dict,
    count_tokens,
    load_model,
    make_image_batch,
    mean,
    resolve_tokens_after,
    sync_if_cuda,
)
from wall_x.data.utils import KEY_MAPPINGS


def make_image_items(args):
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    image_key = args.image_key
    if image_key is None:
        image_key = next(iter(KEY_MAPPINGS[args.repo_id]["camera"].keys()))

    image_indices = np.linspace(0, max(0, len(dataset) - 1), args.num_images, dtype=int)
    items = []
    for idx in image_indices:
        sample = dataset[int(idx)]
        items.append(
            {
                "sample": sample,
                "image_key": image_key,
                "source": f"dataset_index={int(idx)} image_key={image_key}",
            }
        )
    return items


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_step_count(args, model, items, ode_steps: int):
    records = []
    timing_counts = {}
    num_inference_timesteps = ode_steps + 1

    for idx, item in enumerate(items, start=1):
        sync_if_cuda(args.device)
        prepare_start = time.perf_counter()
        batch = make_image_batch(model, item["sample"], item["image_key"], args)
        sync_if_cuda(args.device)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        tokens_before = count_tokens(model, batch, "image")
        tokens_after = resolve_tokens_after(model, batch, "image")

        call_kwargs = dict(
            **batch,
            action_dim=args.action_dim,
            action_horizon=args.pred_horizon,
            mode="predict",
            predict_mode="diffusion",
            unnorm=False,
            num_inference_timesteps=num_inference_timesteps,
        )

        for warmup_idx in range(args.warmup):
            seed = args.base_seed + idx * 1000 - args.warmup + warmup_idx
            set_seed(seed, args.device)
            _ = model(**call_kwargs, profile_timing=False, print_timing=False)

        timing_dicts = []
        last_action = None
        for iter_idx in range(args.iters):
            seed = args.base_seed + idx * 1000 + iter_idx
            set_seed(seed, args.device)
            out = model(**call_kwargs, profile_timing=True, print_timing=False)
            timing_dicts.append(dict(out["timing_results_ms"]))
            if out.get("predict_action") is not None:
                last_action = (
                    out["predict_action"].detach().cpu().to(torch.float32).numpy()
                )

        timing_counts = dict(out.get("timing_counts", timing_counts))
        avg_timing = average_timing_dict(timing_dicts)

        record = {
            "ode_steps": ode_steps,
            "num_inference_timesteps": num_inference_timesteps,
            "index": idx,
            "source": item["source"],
            "external_prepare_batch_ms": prepare_ms,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "predict_action_last": (
                last_action.reshape(-1).tolist() if last_action is not None else None
            ),
            "timings_ms": avg_timing,
        }
        records.append(record)
        print(
            f"[ODE_FIXED] steps={ode_steps} {idx}/{len(items)} "
            f"tokens={tokens_before}->{tokens_after} "
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
        "external_prepare_batch_ms": mean(
            rec["external_prepare_batch_ms"] for rec in records
        ),
        "timings_ms": {
            key: mean(rec["timings_ms"].get(key, 0.0) for rec in records)
            for key in timing_keys
        },
    }


def action_error(reference_records, candidate_records):
    rows = []
    for ref, cand in zip(reference_records, candidate_records):
        ref_action = np.asarray(ref["predict_action_last"], dtype=np.float32)
        cand_action = np.asarray(cand["predict_action_last"], dtype=np.float32)
        diff = cand_action - ref_action
        rows.append(
            {
                "index": ref["index"],
                "source": ref["source"],
                "action_mae": float(np.mean(np.abs(diff))),
                "action_rmse": float(np.sqrt(np.mean(diff**2))),
                "action_max_abs": float(np.max(np.abs(diff))),
            }
        )
    return {
        "rows": rows,
        "mae": mean(row["action_mae"] for row in rows),
        "rmse": mean(row["action_rmse"] for row in rows),
        "max_abs": mean(row["action_max_abs"] for row in rows),
    }


def append_stage_table(report, baseline_summary, summaries):
    keys = [
        "total_time",
        "external_prepare_batch_ms",
        "embed_processing",
        "image_path_total",
        "vision_image_forward",
        "position_encoding",
        "action_initialization",
        "prefetch_forward",
        "prefill_transformer",
        "cache_preprocessing",
        "ode_integration",
        "ode_transformer_total",
        "postprocessing",
    ]
    all_keys = sorted(
        set().union(*(set(summary["timings_ms"]) for summary in summaries.values()))
    )

    report.append("| stage | 9_step_ms | 7_step_ms | 5_step_ms | 3_step_ms |")
    report.append("|---|---:|---:|---:|---:|")
    seen = set()
    for key in keys + all_keys:
        if key in seen:
            continue
        seen.add(key)
        values = []
        for steps in (9, 7, 5, 3):
            summary = summaries[steps]
            if key == "external_prepare_batch_ms":
                values.append(summary["external_prepare_batch_ms"])
            else:
                values.append(summary["timings_ms"].get(key, 0.0))
        if all(value == 0.0 for value in values):
            continue
        report.append(
            f"| `{key}` | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[3]:.3f} |"
        )
    report.append("")


def write_outputs(args, records, counts):
    summaries = {steps: summarize(records[str(steps)]) for steps in (9, 7, 5, 3)}
    baseline_summary = summaries[9]
    errors = {
        steps: action_error(records["9"], records[str(steps)])
        for steps in (7, 5, 3)
    }

    baseline_total = baseline_summary["timings_ms"].get("total_time", 0.0)
    baseline_ode = baseline_summary["timings_ms"].get("ode_integration", 0.0)

    report = [
        "# Wall-X Fixed ODE Step Timing Report\n",
        f"- dataset_root: `{args.dataset_root}`",
        f"- repo_id: `{args.repo_id}`",
        f"- image_key: `{args.image_key}`",
        f"- model_path: `{args.model_path}`",
        f"- num_images: `{args.num_images}`",
        f"- vispruner_enable: `{args.enable_pruning}`",
        f"- keep_ratio: `{args.keep_ratio}`",
        f"- compared_steps: `9 / 7 / 5 / 3`",
        f"- warmup: `{args.warmup}`",
        f"- iters: `{args.iters}`",
        f"- base_seed: `{args.base_seed}`",
        f"- device: `{args.device}`",
        "",
        "## Summary\n",
        "| ODE steps | total_ms | total_delta_pct_vs_9 | ode_ms | ode_delta_pct_vs_9 | action_mae_vs_9 | action_rmse_vs_9 | action_max_abs_vs_9 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for steps in (9, 7, 5, 3):
        summary = summaries[steps]
        total = summary["timings_ms"].get("total_time", 0.0)
        ode = summary["timings_ms"].get("ode_integration", 0.0)
        total_pct = (total - baseline_total) / baseline_total * 100.0 if baseline_total else 0.0
        ode_pct = (ode - baseline_ode) / baseline_ode * 100.0 if baseline_ode else 0.0
        err = errors.get(steps)
        mae = err["mae"] if err else 0.0
        rmse = err["rmse"] if err else 0.0
        max_abs = err["max_abs"] if err else 0.0
        report.append(
            f"| {steps} | {total:.3f} | {total_pct:+.2f}% | {ode:.3f} | {ode_pct:+.2f}% | "
            f"{mae:.6f} | {rmse:.6f} | {max_abs:.6f} |"
        )

    report.append("")
    report.append("## Stage Timing\n")
    append_stage_table(report, baseline_summary, summaries)

    report.append("## Per-Sample Action Error\n")
    for steps in (7, 5, 3):
        report.append(f"### {steps} steps vs 9 steps\n")
        report.append("| idx | source | action_mae | action_rmse | action_max_abs |")
        report.append("|---:|---|---:|---:|---:|")
        for row in errors[steps]["rows"]:
            report.append(
                f"| {row['index']} | `{row['source']}` | "
                f"{row['action_mae']:.6f} | {row['action_rmse']:.6f} | {row['action_max_abs']:.6f} |"
            )
        report.append("")

    report.append("## Raw Results\n")
    report.append(f"- `{args.results_json}`")

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text("\n".join(report), encoding="utf-8")
    Path(args.results_json).write_text(
        json.dumps(
            {
                "args": vars(args),
                "records": records,
                "summaries": summaries,
                "action_errors_vs_9": errors,
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
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/wall_x/datasheet/libero_all")
    parser.add_argument("--repo-id", default="libero_all")
    parser.add_argument("--image-key", default="observation.images.faceImg")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--enable-pruning", action="store_true", default=True)
    parser.add_argument("--disable-pruning", dest="enable_pruning", action="store_false")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--report-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_fixed_steps_9_7_5_3_report.md",
    )
    parser.add_argument(
        "--results-json",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_fixed_steps_9_7_5_3_results.json",
    )
    args = parser.parse_args()

    items = make_image_items(args)
    model = load_model(args, args.enable_pruning)

    records = {}
    counts = {}
    for steps in (9, 7, 5, 3):
        records[str(steps)], counts[str(steps)] = run_step_count(
            args, model, items, steps
        )

    write_outputs(args, records, counts)
    print(f"[ODE_FIXED] report={args.report_path}", flush=True)
    print(f"[ODE_FIXED] results_json={args.results_json}", flush=True)

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
