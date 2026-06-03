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

from scripts.profile_ode_fixed_steps_lerobot_images import make_image_items, set_seed
from scripts.profile_vispruner_lerobot_media import (
    average_timing_dict,
    count_tokens,
    load_model,
    make_image_batch,
    mean,
    resolve_tokens_after,
    sync_if_cuda,
)


CASE_SPECS = {
    "fixed_10": {
        "label": "Fixed 10 updates",
        "early_stop_enable": False,
        "cache_enable": False,
    },
    "cache_i2": {
        "label": "V5 ODE cache interval 2",
        "early_stop_enable": False,
        "cache_enable": True,
        "interval_arg": "cache_interval_2",
        "start_step_arg": "cache_start_step",
    },
    "cache_i3": {
        "label": "V5 ODE cache interval 3",
        "early_stop_enable": False,
        "cache_enable": True,
        "interval_arg": "cache_interval_3",
        "start_step_arg": "cache_start_step",
    },
    "early_tradeoff": {
        "label": "V3 early stop tradeoff",
        "early_stop_enable": True,
        "cache_enable": False,
        "threshold_arg": "tradeoff_threshold",
        "min_steps_arg": "tradeoff_min_steps",
        "patience_arg": "tradeoff_patience",
        "metric": "mean_abs",
    },
}


def case_runtime_kwargs(args, case_name):
    spec = CASE_SPECS[case_name]
    kwargs = {
        "ode_early_stop_enable": False,
        "ode_cache_enable": bool(spec.get("cache_enable", False)),
    }
    if spec.get("cache_enable", False):
        kwargs.update(
            {
                "ode_cache_interval": int(getattr(args, spec["interval_arg"])),
                "ode_cache_start_step": int(getattr(args, spec["start_step_arg"])),
            }
        )
    if spec.get("early_stop_enable", False):
        kwargs.update(
            {
                "ode_early_stop_enable": True,
                "ode_early_stop_threshold": float(getattr(args, spec["threshold_arg"])),
                "ode_early_stop_min_steps": int(getattr(args, spec["min_steps_arg"])),
                "ode_early_stop_patience": int(getattr(args, spec["patience_arg"])),
                "ode_early_stop_metric": spec["metric"],
            }
        )
    return kwargs


@torch.no_grad()
def run_case(args, model, items, case_name):
    records = []
    timing_counts = {}
    runtime_kwargs = case_runtime_kwargs(args, case_name)

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
            f"[V5_CACHE] {case_name} {idx}/{len(items)} "
            f"tokens={tokens_before}->{tokens_after} "
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
        "max_sample_max_abs": max(row["action_max_abs"] for row in rows),
    }


def pct_delta(value, base):
    return (value - base) / base * 100.0 if base else 0.0


def append_stage_table(report, summaries):
    cases = list(CASE_SPECS)
    priority = [
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
        "# Wall-X V5 ODE Cache Image Dataset Report\n",
        f"- dataset_root: `{args.dataset_root}`",
        f"- repo_id: `{args.repo_id}`",
        f"- image_key: `{args.image_key}`",
        f"- model_path: `{args.model_path}`",
        f"- num_images: `{args.num_images}`",
        f"- vispruner_enable: `{args.enable_pruning}`",
        f"- keep_ratio: `{args.keep_ratio}`",
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
        "| case | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | cache_refreshes | cache_hits | cache_hit_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in CASE_SPECS:
        summary = summaries[case]
        total = summary["timings_ms"].get("total_time", 0.0)
        ode = summary["timings_ms"].get("ode_integration", 0.0)
        err = errors.get(case, {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0})
        report.append(
            f"| `{case}` | {total:.3f} | {pct_delta(total, fixed_total):+.2f}% | "
            f"{ode:.3f} | {pct_delta(ode, fixed_ode):+.2f}% | "
            f"{summary['actual_steps_mean']:.2f} | "
            f"{summary['cache_refreshes_mean']:.2f} | {summary['cache_hits_mean']:.2f} | "
            f"{summary['cache_hit_rate_mean']:.2%} | "
            f"{err['mae']:.6f} | {err['rmse']:.6f} | {err['max_abs']:.6f} |"
        )

    report.extend([
        "",
        "## Interpretation\n",
        "- `fixed_10` is the exact no-cache baseline with the original 10 Euler updates.",
        "- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.",
        "- Accuracy is reported as action difference against `fixed_10` under the same sample and seed.",
        "- Fine-grained timings use `profile_timing=True`, so absolute numbers include CUDA event synchronization overhead; paired deltas are the useful signal.",
        "",
        "## Stage Timing\n",
    ])
    append_stage_table(report, summaries)

    report.extend([
        "## Per-Sample Paired Results\n",
        "| idx | source | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    case_errors = {case: {row["index"]: row for row in err["rows"]} for case, err in errors.items()}
    for fixed_rec, cache2_rec, cache3_rec, trade_rec in zip(
        records["fixed_10"], records["cache_i2"], records["cache_i3"], records["early_tradeoff"]
    ):
        idx = fixed_rec["index"]
        report.append(
            f"| {idx} | `{fixed_rec['source']}` | "
            f"{fixed_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{cache2_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{cache3_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{trade_rec['timings_ms'].get('total_time', 0.0):.3f} | "
            f"{case_errors['cache_i2'][idx]['action_mae']:.6f} | "
            f"{case_errors['cache_i3'][idx]['action_mae']:.6f} | "
            f"{case_errors['early_tradeoff'][idx]['action_mae']:.6f} |"
        )

    report.extend([
        "",
        "## Raw Results\n",
        f"- `{args.results_json}`",
    ])

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
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/wall_x/datasheet/libero_all")
    parser.add_argument("--repo-id", default="libero_all")
    parser.add_argument("--image-key", default="observation.images.faceImg")
    parser.add_argument("--num-images", type=int, default=50)
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
        default="/root/autodl-tmp/wall_x/workspace/v5_ode_cache/libero_50img_v5_ode_cache_report.md",
    )
    parser.add_argument(
        "--results-json",
        default="/root/autodl-tmp/wall_x/workspace/v5_ode_cache/libero_50img_v5_ode_cache_results.json",
    )
    args = parser.parse_args()

    items = make_image_items(args)
    model = load_model(args, args.enable_pruning)

    records = {}
    counts = {}
    for case_name in CASE_SPECS:
        records[case_name], counts[case_name] = run_case(args, model, items, case_name)

    write_outputs(args, records, counts)
    print(f"[V5_CACHE] report={args.report_path}", flush=True)
    print(f"[V5_CACHE] results_json={args.results_json}", flush=True)

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
