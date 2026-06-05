import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.profile_vispruner_lerobot_media import (
    count_tokens,
    load_model,
    make_image_batch,
    make_items,
    mean,
    resolve_tokens_after,
    sync_if_cuda,
)


def set_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_case(args, case_name: str, enable_pruning: bool, items):
    model = load_model(args, enable_pruning)
    records = []

    for idx, item in enumerate(items, start=1):
        batch = make_image_batch(model, item["sample"], args.image_key, args)
        tokens_before = count_tokens(model, batch, "image")
        tokens_after = resolve_tokens_after(model, batch, "image")

        call_kwargs = dict(
            **batch,
            action_dim=args.action_dim,
            action_horizon=args.pred_horizon,
            num_inference_timesteps=args.num_inference_timesteps,
            mode="predict",
            predict_mode="diffusion",
            unnorm=args.unnorm,
        )

        for warm_idx in range(args.warmup):
            set_seed(args.seed + idx * 1009 + warm_idx, args.device)
            _ = model(**call_kwargs, profile_timing=False, print_timing=False)
            sync_if_cuda(args.device)

        set_seed(args.seed + idx, args.device)
        out = model(
            **call_kwargs,
            profile_timing=args.profile_timing,
            print_timing=False,
        )
        sync_if_cuda(args.device)
        action = out["predict_action"].detach().cpu().float()
        record = {
            "case_name": case_name,
            "sample_index": idx,
            "source": item["source"],
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "timings_ms": dict(out.get("timing_results_ms", {})),
        }
        if args.save_actions:
            record["predict_action"] = action.tolist()
        records.append((record, action))

        if idx == 1 or idx == len(items) or idx % max(1, args.progress_interval) == 0:
            total_ms = record["timings_ms"].get("total_time", 0.0)
            print(
                f"[ACTION_COMPARE] {case_name} {idx}/{len(items)} "
                f"tokens={tokens_before}->{tokens_after} total_ms={total_ms:.3f}",
                flush=True,
            )

        del batch, out
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def compare_actions(baseline_records, pruned_records, args):
    paired = []
    for (base_rec, base_action), (pruned_rec, pruned_action) in zip(
        baseline_records, pruned_records
    ):
        diff = pruned_action - base_action
        flat_base = base_action.reshape(1, -1)
        flat_pruned = pruned_action.reshape(1, -1)
        cosine = F.cosine_similarity(flat_base, flat_pruned).item()
        mae = diff.abs().mean().item()
        rmse = torch.sqrt((diff * diff).mean()).item()
        max_abs = diff.abs().max().item()
        allclose = bool(
            torch.allclose(
                pruned_action,
                base_action,
                rtol=args.rtol,
                atol=args.atol,
            )
        )
        paired.append(
            {
                "sample_index": base_rec["sample_index"],
                "source": base_rec["source"],
                "baseline_tokens_after": base_rec["tokens_after"],
                "pruned_tokens_after": pruned_rec["tokens_after"],
                "mae": mae,
                "rmse": rmse,
                "max_abs": max_abs,
                "cosine_similarity": cosine,
                "allclose": allclose,
                "baseline_total_time_ms": base_rec["timings_ms"].get("total_time", 0.0),
                "pruned_total_time_ms": pruned_rec["timings_ms"].get("total_time", 0.0),
            }
        )
    return paired


def summarize_comparisons(rows):
    return {
        "num_samples": len(rows),
        "mean_mae": mean(row["mae"] for row in rows),
        "mean_rmse": mean(row["rmse"] for row in rows),
        "mean_max_abs": mean(row["max_abs"] for row in rows),
        "max_max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "mean_cosine_similarity": mean(row["cosine_similarity"] for row in rows),
        "allclose_rate": mean(1.0 if row["allclose"] else 0.0 for row in rows),
        "baseline_tokens_after": mean(row["baseline_tokens_after"] for row in rows),
        "pruned_tokens_after": mean(row["pruned_tokens_after"] for row in rows),
        "baseline_total_time_ms": mean(row["baseline_total_time_ms"] for row in rows),
        "pruned_total_time_ms": mean(row["pruned_total_time_ms"] for row in rows),
    }


def write_outputs(args, baseline_records, pruned_records, paired_rows):
    summary = summarize_comparisons(paired_rows)
    output = {
        "args": vars(args),
        "summary": summary,
        "paired_action_metrics": paired_rows,
        "records": {
            "baseline": [record for record, _ in baseline_records],
            "pruned": [record for record, _ in pruned_records],
        },
    }
    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    total_delta = summary["pruned_total_time_ms"] - summary["baseline_total_time_ms"]
    total_pct = (
        total_delta / summary["baseline_total_time_ms"] * 100.0
        if summary["baseline_total_time_ms"]
        else 0.0
    )
    lines = [
        "# Wall-X Original vs Predictor Action Output Comparison",
        "",
        f"- samples: `{summary['num_samples']}`",
        f"- pruned_strategy: `{args.pruned_strategy}`",
        f"- predictor_checkpoint: `{args.predictor_checkpoint}`",
        f"- predictor_source: `{args.predictor_source}`",
        f"- predictor_early_layer: `{args.predictor_early_layer}`",
        f"- seed: `{args.seed}`",
        f"- allclose atol/rtol: `{args.atol}` / `{args.rtol}`",
        "",
        "## Summary",
        "",
        f"- tokens: baseline `{summary['baseline_tokens_after']:.2f}`, pruned `{summary['pruned_tokens_after']:.2f}`",
        f"- total_time_ms: baseline `{summary['baseline_total_time_ms']:.3f}`, pruned `{summary['pruned_total_time_ms']:.3f}`, delta `{total_delta:+.3f}` (`{total_pct:+.2f}%`)",
        f"- action MAE: `{summary['mean_mae']:.6f}`",
        f"- action RMSE: `{summary['mean_rmse']:.6f}`",
        f"- action mean max_abs: `{summary['mean_max_abs']:.6f}`",
        f"- action worst max_abs: `{summary['max_max_abs']:.6f}`",
        f"- action cosine_similarity: `{summary['mean_cosine_similarity']:.6f}`",
        f"- action allclose_rate: `{summary['allclose_rate']:.4f}`",
        "",
        "## Metric Meaning",
        "",
        "| metric | meaning |",
        "|---|---|",
        "| `MAE` | Mean absolute difference between predictor and original action tensors. Lower is closer. |",
        "| `RMSE` | Root mean squared difference between action tensors. Lower is closer. |",
        "| `max_abs` | Maximum absolute element-wise action difference per sample. Lower is closer. |",
        "| `cosine_similarity` | Direction similarity between flattened action tensors. Closer to 1 is better. |",
        "| `allclose_rate` | Fraction of samples passing `torch.allclose` under the configured atol/rtol. |",
    ]
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/wall_x/pretrained/wall-oss-fast")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/wall_x/datasheet/libero_all")
    parser.add_argument("--repo-id", default="libero_all")
    parser.add_argument("--image-key", default="observation.images.faceImg")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument(
        "--pruned-strategy",
        default="predictor_early",
        choices=["predictor_score", "predictor_early", "topk_attention", "norm"],
    )
    parser.add_argument(
        "--predictor-checkpoint",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt",
    )
    parser.add_argument(
        "--predictor-source",
        default="early_hidden",
        choices=["image_embeds", "patch_embeds", "early_hidden"],
    )
    parser.add_argument("--predictor-early-layer", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument("--num-inference-timesteps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--unnorm", action="store_true")
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--save-actions", action="store_true")
    parser.add_argument(
        "--report-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_logs/action_compare_original_vs_predictor_early_report.md",
    )
    parser.add_argument(
        "--results-json",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_logs/action_compare_original_vs_predictor_early_results.json",
    )
    args = parser.parse_args()

    image_items, _ = make_items(args)
    baseline_records = run_case(args, "baseline_original", False, image_items)
    pruned_records = run_case(args, "predictor_pruned", True, image_items)
    paired_rows = compare_actions(baseline_records, pruned_records, args)
    write_outputs(args, baseline_records, pruned_records, paired_rows)
    print(f"[ACTION_COMPARE] report={args.report_path}", flush=True)
    print(f"[ACTION_COMPARE] results_json={args.results_json}", flush=True)


if __name__ == "__main__":
    main()
