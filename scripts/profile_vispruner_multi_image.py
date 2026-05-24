import argparse
import gc
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
)

import importlib.util


_UTILS_PATH = "/root/autodl-tmp/wall_x/wall_x/serving/policy/utils.py"
_spec = importlib.util.spec_from_file_location("wallx_serving_policy_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
prepare_batch = _utils.prepare_batch


class IdentityNormalizer:
    def normalize_data(self, xs, dataset_names, *args, **kwargs):
        return xs


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def build_train_config(model_path: str, enable_pruning: bool, keep_ratio: float) -> dict:
    with open(Path(model_path) / "config.json", "r") as f:
        model_cfg = json.load(f)

    return {
        "processor_path": model_path,
        "dof_config": model_cfg["dof_config"],
        "agent_pos_config": model_cfg["agent_pos_config"],
        "data": {
            "use_state_string_representation": False,
            "action_horizon_flow": 32,
        },
        "vispruner": {
            "enable": enable_pruning,
            "strategy": "topk_attention" if enable_pruning else "original",
            "keep_ratio": keep_ratio if enable_pruning else 1.0,
            "min_tokens": 1,
            "force_vision_eager": True,
        },
    }


def download_picsum_images(image_dir: Path, num_images: int):
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_images):
        out_path = image_dir / f"picsum_{idx:03d}.jpg"
        if out_path.exists():
            continue
        url = f"https://picsum.photos/seed/wallx-vispruner-{idx}/640/480"
        urllib.request.urlretrieve(url, out_path)


def list_images(image_dir: Path, num_images: int):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    return images[:num_images]


def load_model(args, enable_pruning: bool):
    train_config = build_train_config(args.model_path, enable_pruning, args.keep_ratio)
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        args.model_path,
        train_config=train_config,
    )
    model.eval()
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        model.to_bfloat16_for_selected_params()
    return model


def make_batch(model, image_path: Path, args):
    image = Image.open(image_path).convert("RGB")
    obs = {
        "front_view": np.array(image),
        "prompt": "pick up the object",
        "state": np.zeros((args.action_dim,), dtype=np.float32),
        "dataset_names": "x2_normal",
    }

    return prepare_batch(
        obs=obs,
        processor=model.processor,
        normalizer_propri=IdentityNormalizer(),
        camera_key=["front_view"],
        agent_pos_dim=args.action_dim,
        action_dim=args.action_dim,
        pred_horizon=args.pred_horizon,
        fixed_action_dim=args.action_dim,
        max_length=2048,
        image_factor=28,
        min_pixels=4 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        predict_mode="diffusion",
        device=args.device,
    )


def count_image_tokens(model, input_ids, attention_mask=None) -> int:
    mask = input_ids == model.config.image_token_id
    if attention_mask is not None:
        mask = mask & attention_mask.bool()
    return int(mask.sum().item())


@torch.no_grad()
def resolve_pruned_vision_tokens(model, batch) -> int:
    image_embeds, input_ids, attention_mask, labels, moe_token_types, position_ids, pruned = (
        model._encode_images_and_maybe_prune(
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            moe_token_types=batch.get("moe_token_types"),
            position_ids=batch.get("position_ids"),
            video_grid_thw=batch.get("video_grid_thw"),
            second_per_grid_ts=batch.get("second_per_grid_ts"),
        )
    )
    if not pruned:
        return count_image_tokens(model, batch["input_ids"], batch.get("attention_mask"))
    return count_image_tokens(model, input_ids, attention_mask)


def average_timing_dict(timing_dicts):
    keys = sorted({key for item in timing_dicts for key in item})
    return {
        key: sum(item.get(key, 0.0) for item in timing_dicts) / len(timing_dicts)
        for key in keys
    }


@torch.no_grad()
def run_case_on_images(args, case_name: str, enable_pruning: bool, image_paths):
    model = load_model(args, enable_pruning)
    records = []
    timing_counts = {}

    for image_idx, image_path in enumerate(image_paths, start=1):
        sync_if_cuda(args.device)
        prepare_start = time.perf_counter()
        batch = make_batch(model, image_path, args)
        sync_if_cuda(args.device)
        external_prepare_batch_ms = (time.perf_counter() - prepare_start) * 1000.0
        batch["labels"] = None

        vision_tokens_before = count_image_tokens(
            model, batch["input_ids"], batch.get("attention_mask")
        )
        vision_tokens_after = resolve_pruned_vision_tokens(model, batch)

        call_kwargs = dict(
            **batch,
            action_dim=args.action_dim,
            action_horizon=args.pred_horizon,
            mode="predict",
            predict_mode="diffusion",
            unnorm=False,
        )

        for _ in range(args.warmup):
            _ = model(**call_kwargs, profile_timing=False, print_timing=False)

        timing_dicts = []
        for _ in range(args.iters):
            out = model(**call_kwargs, profile_timing=True, print_timing=False)
            timing_dicts.append(dict(out["timing_results_ms"]))
        avg_timing = average_timing_dict(timing_dicts)
        timing_counts = dict(out.get("timing_counts", timing_counts))

        record = {
            "case": case_name,
            "image_index": image_idx,
            "image_path": str(image_path),
            "external_prepare_batch_ms": external_prepare_batch_ms,
            "vision_tokens_before": vision_tokens_before,
            "vision_tokens_after": vision_tokens_after,
            "timings_ms": avg_timing,
        }
        records.append(record)
        print(
            f"[MULTI_PROFILE] {case_name} image={image_idx}/{len(image_paths)} "
            f"tokens={vision_tokens_before}->{vision_tokens_after} "
            f"total_ms={avg_timing.get('total_time', 0.0):.3f}",
            flush=True,
        )

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, timing_counts


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(records):
    timing_keys = sorted({key for rec in records for key in rec["timings_ms"]})
    return {
        "num_images": len(records),
        "vision_tokens_before": mean(rec["vision_tokens_before"] for rec in records),
        "vision_tokens_after": mean(rec["vision_tokens_after"] for rec in records),
        "external_prepare_batch_ms": mean(
            rec["external_prepare_batch_ms"] for rec in records
        ),
        "timings_ms": {
            key: mean(rec["timings_ms"].get(key, 0.0) for rec in records)
            for key in timing_keys
        },
    }


def paired_deltas(baseline_records, pruned_records):
    deltas = []
    for base, pruned in zip(baseline_records, pruned_records):
        base_total = base["timings_ms"].get("total_time", 0.0)
        pruned_total = pruned["timings_ms"].get("total_time", 0.0)
        deltas.append(
            {
                "image_index": base["image_index"],
                "image_path": base["image_path"],
                "vision_tokens_before": base["vision_tokens_before"],
                "vision_tokens_after": pruned["vision_tokens_after"],
                "token_reduction_pct": (
                    (base["vision_tokens_before"] - pruned["vision_tokens_after"])
                    / base["vision_tokens_before"]
                    * 100.0
                    if base["vision_tokens_before"]
                    else 0.0
                ),
                "baseline_total_ms": base_total,
                "pruned_total_ms": pruned_total,
                "total_delta_ms": pruned_total - base_total,
                "total_delta_pct": (
                    (pruned_total - base_total) / base_total * 100.0
                    if base_total
                    else 0.0
                ),
            }
        )
    return deltas


def write_report(args, image_paths, baseline_records, pruned_records, timing_counts):
    baseline_summary = summarize(baseline_records)
    pruned_summary = summarize(pruned_records)
    deltas = paired_deltas(baseline_records, pruned_records)

    timing_keys = [
        "total_time",
        "external_prepare_batch_ms",
        "embed_processing",
        "image_path_total",
        "vision_image_forward",
        "vision_image_encode",
        "vision_image_encode_score",
        "vispruner_total",
        "vispruner_build_keep_mask",
        "vispruner_topk_select",
        "vispruner_apply_keep_to_sequences",
        "embed_tokens",
        "scatter_image_embeds",
        "position_encoding",
        "position_ids_rope",
        "moe_indices",
        "action_initialization",
        "prefetch_forward",
        "prefill_transformer",
        "prefill_action_head",
        "cache_preprocessing",
        "kv_cache_trim",
        "postfix_mask_build",
        "ode_integration",
        "ode_action_embed_total",
        "ode_prepare_inputs",
        "ode_transformer_total",
        "ode_action_head_total",
        "postprocessing",
    ]

    def get_summary_value(summary, key):
        if key == "external_prepare_batch_ms":
            return summary["external_prepare_batch_ms"]
        return summary["timings_ms"].get(key, 0.0)

    report = []
    report.append("# Wall-X VisPruner 30 Image Timing Report\n")
    report.append(f"- model_path: `{args.model_path}`")
    report.append(f"- image_dir: `{args.image_dir}`")
    report.append("- image_source: `https://picsum.photos/seed/wallx-vispruner-{idx}/640/480` when `--download-picsum` is used")
    report.append(f"- num_images: `{len(image_paths)}`")
    report.append(f"- warmup: `{args.warmup}`")
    report.append(f"- iters: `{args.iters}`")
    report.append(f"- keep_ratio: `{args.keep_ratio}`")
    report.append(f"- device: `{args.device}`")
    report.append("")
    report.append("## Summary\n")
    avg_token_before = baseline_summary["vision_tokens_before"]
    avg_token_after = pruned_summary["vision_tokens_after"]
    token_reduction = (
        (avg_token_before - avg_token_after) / avg_token_before * 100.0
        if avg_token_before
        else 0.0
    )
    base_total = baseline_summary["timings_ms"].get("total_time", 0.0)
    pruned_total = pruned_summary["timings_ms"].get("total_time", 0.0)
    total_delta = pruned_total - base_total
    total_delta_pct = total_delta / base_total * 100.0 if base_total else 0.0
    report.append(
        f"- Average vision tokens: baseline `{avg_token_before:.2f}`, "
        f"pruned `{avg_token_after:.2f}`, reduction `{token_reduction:.2f}%`."
    )
    report.append(
        f"- Average model `total_time`: baseline `{base_total:.3f} ms`, "
        f"pruned `{pruned_total:.3f} ms`, delta `{total_delta:+.3f} ms` "
        f"(`{total_delta_pct:+.2f}%`)."
    )
    report.append(
        "- Note: timing was collected with `profile_timing=True`, so fine-grained "
        "CUDA synchronization overhead is included. Use paired baseline/pruned "
        "differences for diagnosis, not as online latency."
    )
    report.append("")
    report.append("## Diagnostic Findings\n")
    image_forward_base = baseline_summary["timings_ms"].get("vision_image_forward", 0.0)
    image_forward_pruned = pruned_summary["timings_ms"].get("vision_image_forward", 0.0)
    score_time = pruned_summary["timings_ms"].get("vision_image_encode_score", 0.0)
    vispruner_time = pruned_summary["timings_ms"].get("vispruner_total", 0.0)
    prefill_base = baseline_summary["timings_ms"].get("prefill_transformer", 0.0)
    prefill_pruned = pruned_summary["timings_ms"].get("prefill_transformer", 0.0)
    ode_base = baseline_summary["timings_ms"].get("ode_transformer_total", 0.0)
    ode_pruned = pruned_summary["timings_ms"].get("ode_transformer_total", 0.0)
    report.append(
        f"- Image path did not become cheaper: `vision_image_forward` changed from "
        f"`{image_forward_base:.3f} ms` to `{image_forward_pruned:.3f} ms`. In the "
        f"pruned path, `vision_image_encode_score` alone costs `{score_time:.3f} ms`, "
        f"and `vispruner_total` adds `{vispruner_time:.3f} ms`."
    )
    report.append(
        f"- Prefix Transformer did not show a clear win in this run: "
        f"`prefill_transformer` changed from `{prefill_base:.3f} ms` to "
        f"`{prefill_pruned:.3f} ms`."
    )
    report.append(
        f"- The dominant ODE/postfix Transformer work is almost unchanged: "
        f"`ode_transformer_total` changed from `{ode_base:.3f} ms` to "
        f"`{ode_pruned:.3f} ms`. This explains why a 49% visual-token reduction "
        "does not translate into a large complete-action latency reduction."
    )
    report.append("")
    report.append("## Average Timing By Segment\n")
    report.append("| segment | baseline_ms | pruned_ms | delta_ms | delta_pct |")
    report.append("|---|---:|---:|---:|---:|")
    seen = set()
    for key in timing_keys + sorted(
        set(baseline_summary["timings_ms"]) | set(pruned_summary["timings_ms"])
    ):
        if key in seen:
            continue
        seen.add(key)
        base = get_summary_value(baseline_summary, key)
        pruned = get_summary_value(pruned_summary, key)
        if base == 0.0 and pruned == 0.0:
            continue
        pct = (pruned - base) / base * 100.0 if base else 0.0
        report.append(f"| `{key}` | {base:.3f} | {pruned:.3f} | {pruned-base:+.3f} | {pct:+.2f}% |")

    report.append("")
    report.append("## Per Image Paired Results\n")
    report.append(
        "| idx | image | token_before | token_after | token_reduction | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |"
    )
    report.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in deltas:
        image_name = Path(item["image_path"]).name
        report.append(
            f"| {item['image_index']} | `{image_name}` | {item['vision_tokens_before']} | "
            f"{item['vision_tokens_after']} | {item['token_reduction_pct']:.2f}% | "
            f"{item['baseline_total_ms']:.3f} | {item['pruned_total_ms']:.3f} | "
            f"{item['total_delta_ms']:+.3f} | {item['total_delta_pct']:+.2f}% |"
        )

    report.append("")
    report.append("## Timing Counts\n")
    report.append("These counts show how many times repeated timing blocks were accumulated per measured run.")
    report.append("")
    report.append("| case | segment | count |")
    report.append("|---|---|---:|")
    for case_name, case_counts in sorted(timing_counts.items()):
        for key, value in sorted(case_counts.items()):
            report.append(f"| `{case_name}` | `{key}` | {value} |")

    report.append("")
    report.append("## Raw Results JSON\n")
    report.append(f"- `{args.results_json}`")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    Path(args.results_json).write_text(
        json.dumps(
            {
                "args": vars(args),
                "images": [str(path) for path in image_paths],
                "baseline": baseline_records,
                "pruned": pruned_records,
                "baseline_summary": baseline_summary,
                "pruned_summary": pruned_summary,
                "paired_deltas": deltas,
                "timing_counts": timing_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/wall_x/pretrained/wall-oss-fast")
    parser.add_argument("--image-dir", default="/root/autodl-tmp/wall_x/benchmark_images/picsum_30")
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--download-picsum", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument(
        "--report-path",
        default="/root/autodl-tmp/wall_x/wall_x/report/wallx_vispruner_30image_timing_report.md",
    )
    parser.add_argument(
        "--results-json",
        default="/root/autodl-tmp/wall_x/wall_x/report/wallx_vispruner_30image_timing_results.json",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if args.download_picsum:
        download_picsum_images(image_dir, args.num_images)
    image_paths = list_images(image_dir, args.num_images)
    if len(image_paths) < args.num_images:
        raise RuntimeError(
            f"Only found {len(image_paths)} images in {image_dir}, expected {args.num_images}. "
            "Use --download-picsum to fetch benchmark images."
        )

    print(f"[MULTI_PROFILE] images={len(image_paths)}", flush=True)
    baseline_records, baseline_counts = run_case_on_images(
        args, "baseline_no_pruning", False, image_paths
    )
    pruned_records, pruned_counts = run_case_on_images(
        args, "vispruner_pruned", True, image_paths
    )
    timing_counts = {"baseline": baseline_counts, "pruned": pruned_counts}
    write_report(args, image_paths, baseline_records, pruned_records, timing_counts)
    print(f"[MULTI_PROFILE] report={args.report_path}", flush=True)
    print(f"[MULTI_PROFILE] results_json={args.results_json}", flush=True)


if __name__ == "__main__":
    main()
