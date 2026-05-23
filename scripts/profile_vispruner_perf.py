import argparse
import gc
import json
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


def cuda_time_call(fn, device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    import time

    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


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

    batch = make_batch(
        model=model,
        image_path=args.image_path,
        device=args.device,
        action_dim=args.action_dim,
        pred_horizon=args.pred_horizon,
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

        timings = []
        for _ in range(args.iters):
            elapsed_ms = cuda_time_call(
                lambda: model(**call_kwargs, profile_timing=False, print_timing=False),
                args.device,
            )
            timings.append(elapsed_ms)

    avg_ms = sum(timings) / len(timings)
    print(f"[VISPRUNER_PROFILE] last_total_time_ms={timings[-1]:.3f}", flush=True)
    print(
        f"[VISPRUNER_PROFILE] average_total_time_ms={avg_ms:.3f} over {args.iters} runs",
        flush=True,
    )

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    if args.case in ("baseline", "both"):
        run_case(args, "baseline_no_pruning", enable_pruning=False)
    if args.case in ("pruned", "both"):
        run_case(args, "vispruner_pruned", enable_pruning=True)


if __name__ == "__main__":
    main()
