import argparse
import time

import torch
import yaml
from PIL import Image

from wall_x.data.utils import preprocesser_call
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
)


def build_prompt(action_horizon: int) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Observation: front view: <|vision_start|><|image_pad|><|vision_end|>\n"
        "Instruction: pick the red block.\n"
        "Predict the next action in robot action.\n"
        "Proprioception: <|propri|>\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        + "<|action|>" * action_horizon
        + "<|im_end|>\n"
    )


def print_tensor_shape(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        print(f"[ShapeCheck] {name}: shape={tuple(value.shape)} dtype={value.dtype}")
    else:
        print(f"[ShapeCheck] {name}: {type(value).__name__}")


def main() -> None:
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
    parser.add_argument("--dataset-name", default="x2_normal")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument(
        "--train-config",
        default=None,
        help="Optional Wall-X training YAML. Use this to test config-driven switches such as vispruner.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    image = Image.open(args.image_path).convert("RGB")

    train_config = None
    if args.train_config is not None:
        with open(args.train_config, "r") as f:
            train_config = yaml.safe_load(f)
        train_config.setdefault("processor_path", args.model_path)

    model = Qwen2_5_VLMoEForAction.from_pretrained(
        args.model_path, train_config=train_config
    )
    model.eval()
    model = model.to(args.device)
    model = model.bfloat16()
    processor = model.processor

    print("[ShapeCheck] ===== model config =====")
    print(
        f"[ShapeCheck] vispruner_enable={getattr(model.config, 'vispruner_enable', False)}"
    )
    print(
        f"[ShapeCheck] vispruner_strategy={getattr(model.config, 'vispruner_strategy', 'original')}"
    )
    print(
        f"[ShapeCheck] vispruner_keep_ratio={getattr(model.config, 'vispruner_keep_ratio', 1.0)}"
    )
    print(
        f"[ShapeCheck] pruner_enabled={getattr(model, 'vispruner', None).enabled if hasattr(model, 'vispruner') else False}"
    )

    inputs = preprocesser_call(
        processor=processor,
        text=[build_prompt(args.action_horizon)],
        images=[image],
        videos=None,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=2048,
    )
    # This smoke test focuses on shape alignment through the vision/action path.
    # The repo's current channel-loss branch expects trainer-only bookkeeping.
    inputs["labels"] = None

    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    propri_token_id = processor.tokenizer.convert_tokens_to_ids("<|propri|>")
    image_token_id = model.config.image_token_id

    input_ids = inputs["input_ids"]
    image_token_count = int((input_ids == image_token_id).sum().item())
    propri_token_count = int((input_ids == propri_token_id).sum().item())
    action_token_count = int((input_ids == action_token_id).sum().item())

    batch_size = input_ids.shape[0]
    proprioception = torch.randn(
        (batch_size, 1, args.action_dim), dtype=torch.bfloat16
    )
    agent_pos_mask = torch.ones_like(proprioception)
    action_chunk = torch.randn(
        (batch_size, args.action_horizon, args.action_dim), dtype=torch.bfloat16
    )
    dof_mask = torch.ones_like(action_chunk)

    inputs["moe_token_types"] = input_ids == action_token_id
    inputs["proprioception"] = proprioception
    inputs["agent_pos_mask"] = agent_pos_mask
    inputs["action_chunk"] = action_chunk
    inputs["dof_mask"] = dof_mask
    inputs["dataset_names"] = [args.dataset_name] * batch_size

    print("[ShapeCheck] ===== inputs before device move =====")
    for key in [
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
        "moe_token_types",
        "proprioception",
        "agent_pos_mask",
        "action_chunk",
        "dof_mask",
    ]:
        print_tensor_shape(key, inputs.get(key))

    print(f"[ShapeCheck] seq_len={input_ids.shape[1]}")
    print(f"[ShapeCheck] image_token_count={image_token_count}")
    print(f"[ShapeCheck] propri_token_count={propri_token_count}")
    print(f"[ShapeCheck] action_token_count={action_token_count}")

    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    assert inputs["input_ids"].shape == inputs["moe_token_types"].shape
    if inputs["labels"] is not None:
        assert inputs["labels"].shape == inputs["input_ids"].shape
    assert image_token_count > 0
    assert propri_token_count == batch_size
    assert action_token_count == batch_size * args.action_horizon
    assert inputs["proprioception"].shape == inputs["agent_pos_mask"].shape
    assert inputs["action_chunk"].shape == inputs["dof_mask"].shape

    inputs = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    torch.cuda.reset_peak_memory_stats(args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs, mode="validate")
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_mem_gb = (
        torch.cuda.max_memory_allocated(args.device) / 1024**3
        if args.device == "cuda"
        else 0.0
    )

    print("[ShapeCheck] ===== outputs =====")
    print_tensor_shape("logits", outputs.logits)
    print(f"[ShapeCheck] output_seq_len={outputs.logits.shape[1]}")
    print(f"[ShapeCheck] forward_time_ms={elapsed_ms:.3f}")
    print(f"[ShapeCheck] peak_memory_gb={peak_mem_gb:.3f}")
    print(f"[ShapeCheck] logits_has_nan={torch.isnan(outputs.logits).any().item()}")
    print(f"[ShapeCheck] logits_has_inf={torch.isinf(outputs.logits).any().item()}")
    print("[ShapeCheck] PASS")


if __name__ == "__main__":
    main()
