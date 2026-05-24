import argparse
import gc
import json
import time
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


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


class IdentityNormalizer:
    def normalize_data(self, xs, dataset_names, *args, **kwargs):
        return xs


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


def make_batch(model, image_path: str, device: str, action_dim: int, pred_horizon: int):
    image = Image.open(image_path).convert("RGB")
    obs = {
        "front_view": np.array(image),
        "prompt": "pick up the object",
        "state": np.zeros((action_dim,), dtype=np.float32),
        "dataset_names": "x2_normal",
    }

    return prepare_batch(
        obs=obs,
        processor=model.processor,
        normalizer_propri=IdentityNormalizer(),
        camera_key=["front_view"],
        agent_pos_dim=action_dim,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        fixed_action_dim=action_dim,
        max_length=2048,
        image_factor=28,
        min_pixels=4 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        predict_mode="diffusion",
        device=device,
    )


def average_timing_dict(timing_dicts):
    keys = sorted({key for item in timing_dicts for key in item})
    return {
        key: sum(item.get(key, 0.0) for item in timing_dicts) / len(timing_dicts)
        for key in keys
    }


def print_timing_breakdown(case_name: str, averages: dict, last: dict):
    print(f"[VISPRUNER_TIMING] ===== {case_name} average breakdown =====", flush=True)
    for key in sorted(averages):
        print(
            f"[VISPRUNER_TIMING] {case_name}.{key}.avg_ms={averages[key]:.3f}",
            flush=True,
        )
    print(f"[VISPRUNER_TIMING] ===== {case_name} last breakdown =====", flush=True)
    for key in sorted(last):
        print(
            f"[VISPRUNER_TIMING] {case_name}.{key}.last_ms={last[key]:.3f}",
            flush=True,
        )


def print_timing_comparison(baseline: dict, pruned: dict):
    if not baseline or not pruned:
        return
    print("\n[VISPRUNER_COMPARE] ===== segmented timing comparison =====", flush=True)
    print(
        "[VISPRUNER_COMPARE] segment | baseline_avg_ms | pruned_avg_ms | delta_ms | delta_pct",
        flush=True,
    )
    preferred_order = [
        "total_time",
        "embed_processing",
        "vision_image_forward",
        "vision_video_forward",
        "position_encoding",
        "action_initialization",
        "prefetch_forward",
        "cache_preprocessing",
        "ode_integration",
        "postprocessing",
    ]
    keys = [key for key in preferred_order if key in baseline or key in pruned]
    keys.extend(sorted((set(baseline) | set(pruned)) - set(keys)))
    for key in keys:
        base = baseline.get(key, 0.0)
        prun = pruned.get(key, 0.0)
        delta = prun - base
        pct = (delta / base * 100.0) if base else 0.0
        print(
            f"[VISPRUNER_COMPARE] {key} | {base:.3f} | {prun:.3f} | {delta:+.3f} | {pct:+.2f}%",
            flush=True,
        )


def run_case(args, case_name: str, enable_pruning: bool):
    print(f"\n[VISPRUNER_PROFILE] ===== {case_name} =====", flush=True)
    train_config = build_train_config(args.model_path, enable_pruning, args.keep_ratio)
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        args.model_path,
        train_config=train_config,
    )
    model.eval()
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        model.to_bfloat16_for_selected_params()

    print(
        f"[VISPRUNER_PROFILE] vispruner_enable={getattr(model.config, 'vispruner_enable', False)}",
        flush=True,
    )
    print(
        f"[VISPRUNER_PROFILE] vispruner_strategy={getattr(model.config, 'vispruner_strategy', 'original')}",
        flush=True,
    )
    print(
        f"[VISPRUNER_PROFILE] vispruner_keep_ratio={getattr(model.config, 'vispruner_keep_ratio', 1.0)}",
        flush=True,
    )
    print(
        f"[VISPRUNER_PROFILE] pruner_enabled={model.vispruner.enabled}",
        flush=True,
    )

    sync_if_cuda(args.device)
    prepare_start = time.perf_counter()
    batch = make_batch(
        model=model,
        image_path=args.image_path,
        device=args.device,
        action_dim=args.action_dim,
        pred_horizon=args.pred_horizon,
    )
    sync_if_cuda(args.device)
    external_prepare_batch_ms = (time.perf_counter() - prepare_start) * 1000.0
    print(
        f"[VISPRUNER_TIMING] {case_name}.external_prepare_batch_ms={external_prepare_batch_ms:.3f}",
        flush=True,
    )
    batch["labels"] = None

    vision_tokens_before = count_image_tokens(
        model, batch["input_ids"], batch.get("attention_mask")
    )
    vision_tokens_after = resolve_pruned_vision_tokens(model, batch)
    print(
        f"[VISPRUNER_PROFILE] real_vision_tokens_before={vision_tokens_before}",
        flush=True,
    )
    print(
        f"[VISPRUNER_PROFILE] real_vision_tokens_after={vision_tokens_after}",
        flush=True,
    )

    model._log_complexity_track(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pruned_vision_token_count=vision_tokens_after,
    )

    call_kwargs = dict(
        **batch,
        action_dim=args.action_dim,
        action_horizon=args.pred_horizon,
        mode="predict",
        predict_mode="diffusion",
        unnorm=False,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(**call_kwargs, profile_timing=False, print_timing=False)

        timing_dicts = []
        for _ in range(args.iters):
            out = model(**call_kwargs, profile_timing=True, print_timing=False)
            timing_dicts.append(dict(out["timing_results_ms"]))
        timing_counts = dict(out.get("timing_counts", {}))

    avg_timing = average_timing_dict(timing_dicts)
    last_timing = timing_dicts[-1]
    print(
        f"[VISPRUNER_PROFILE] last_total_time_ms={last_timing.get('total_time', 0.0):.3f}",
        flush=True,
    )
    print(
        f"[VISPRUNER_PROFILE] average_total_time_ms={avg_timing.get('total_time', 0.0):.3f} over {args.iters} runs",
        flush=True,
    )
    print_timing_breakdown(case_name, avg_timing, last_timing)
    if timing_counts:
        print(f"[VISPRUNER_TIMING] ===== {case_name} timing counts =====", flush=True)
        for key in sorted(timing_counts):
            print(
                f"[VISPRUNER_TIMING] {case_name}.{key}.count={timing_counts[key]}",
                flush=True,
            )

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return avg_timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/wall_x/pretrained/wall-oss-fast",
    )
    parser.add_argument(
        "--image-path",
        default="/root/autodl-tmp/wall_x/assets/cot_example_frame.png",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument(
        "--case",
        choices=["baseline", "pruned", "both"],
        default="both",
    )
    args = parser.parse_args()

    print(f"[VISPRUNER_PROFILE] image_path={args.image_path}", flush=True)
    print(f"[VISPRUNER_PROFILE] warmup={args.warmup}", flush=True)
    print(f"[VISPRUNER_PROFILE] iters={args.iters}", flush=True)
    print(f"[VISPRUNER_PROFILE] keep_ratio={args.keep_ratio}", flush=True)

    baseline_timing = None
    pruned_timing = None
    if args.case in ("baseline", "both"):
        baseline_timing = run_case(args, "baseline_no_pruning", enable_pruning=False)
    if args.case in ("pruned", "both"):
        pruned_timing = run_case(args, "vispruner_pruned", enable_pruning=True)
    print_timing_comparison(baseline_timing, pruned_timing)


if __name__ == "__main__":
    main()
