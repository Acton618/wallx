import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from wall_x.data.utils import KEY_MAPPINGS
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
)


def build_train_config(model_path: str, keep_ratio: float) -> dict:
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
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
            "enable": True,
            "strategy": "topk_attention",
            "keep_ratio": keep_ratio,
            "min_tokens": 1,
            "force_vision_eager": True,
        },
    }


def load_model(args):
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        args.model_path,
        train_config=build_train_config(args.model_path, args.keep_ratio),
    )
    model.eval()
    model = model.to(args.device)
    if str(args.device).startswith("cuda"):
        model.to_bfloat16_for_selected_params()
    return model


def sample_to_pil(sample, image_key: str) -> Image.Image:
    image = sample[image_key]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = (image.clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(image).convert("RGB")


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def choose_indices(
    dataset_size: int,
    num_samples: int,
    start_index: int = 0,
    end_index: int = None,
) -> list[int]:
    start_index = max(0, int(start_index))
    stop = dataset_size if end_index is None else min(dataset_size, int(end_index))
    if start_index >= stop:
        raise ValueError(
            f"Invalid index range: start_index={start_index}, end_index={stop}."
        )
    available = list(range(start_index, stop))
    if num_samples <= 0 or num_samples >= len(available):
        return available
    positions = np.linspace(0, len(available) - 1, num_samples, dtype=int).tolist()
    return [available[pos] for pos in positions]


def cast_float_tensor(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if dtype_name == "float16":
        return tensor.to(torch.float16)
    if dtype_name == "bfloat16":
        return tensor.to(torch.bfloat16)
    if dtype_name == "float32":
        return tensor.float()
    raise ValueError(f"Unsupported feature dtype: {dtype_name}")


def save_records(records, output_path: Path, meta: dict, shard_idx: int = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "records": records}, output_path)
    label = f" shard={shard_idx}" if shard_idx is not None else ""
    print(f"[TEACHER] saved{label} {len(records)} records to {output_path}", flush=True)


def load_existing_manifest(output_path: Path):
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = sorted(output_path.glob("teacher_scores_shard_*.pt"))
    return {
        "meta": {},
        "shards": [{"path": str(path), "records": None} for path in shards],
    }


@torch.no_grad()
def collect_records(args):
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    image_key = args.image_key
    if image_key is None:
        image_key = next(iter(KEY_MAPPINGS[args.repo_id]["camera"].keys()))

    model = load_model(args)
    records = []
    indices = choose_indices(
        len(dataset),
        args.num_samples,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    meta = {
        "repo_id": args.repo_id,
        "dataset_root": args.dataset_root,
        "image_key": image_key,
        "model_path": args.model_path,
        "strategy": "topk_attention",
        "keep_ratio": args.keep_ratio,
        "feature_source": args.feature_source,
        "early_layer": args.early_layer,
        "feature_dtype": args.feature_dtype,
        "save_image_embeds": args.save_image_embeds,
        "dataset_size": len(dataset),
        "num_records": len(indices),
    }
    output_path = Path(args.output_path)
    if args.shard_size > 0 and args.append_shards and output_path.exists():
        existing_manifest = load_existing_manifest(output_path)
        manifest = existing_manifest.get("shards", [])
    else:
        manifest = []

    for out_idx, dataset_idx in enumerate(indices, start=1):
        sample = dataset[int(dataset_idx)]
        image = sample_to_pil(sample, image_key)
        image_inputs = model.processor.image_processor(
            images=[[image]], return_tensors="pt"
        ).to(args.device)

        pixel_values = image_inputs["pixel_values"].type(model.visual.dtype)
        image_grid_thw = image_inputs["image_grid_thw"]

        sync_if_cuda(args.device)
        start = time.perf_counter()
        image_embeds, image_scores, predictor_features = model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            output_attentions=True,
            return_vispruner_features=True,
            vispruner_feature_source=args.feature_source,
            vispruner_early_layer=args.early_layer,
        )
        sync_if_cuda(args.device)
        vision_score_ms = (time.perf_counter() - start) * 1000.0

        image_lengths = model.vispruner._image_token_lengths(
            image_grid_thw,
            model.config.vision_config.spatial_merge_size,
        ).to(image_embeds.device)

        sync_if_cuda(args.device)
        start = time.perf_counter()
        keep_mask, keep_indices = model.vispruner._build_image_keep_mask(
            image_embeds=image_embeds,
            image_scores=image_scores,
            image_lengths=image_lengths,
        )
        sync_if_cuda(args.device)
        keep_mask_ms = (time.perf_counter() - start) * 1000.0

        record = {
            "dataset_index": int(dataset_idx),
            "image_key": image_key,
            "image_grid_thw": image_grid_thw.detach().cpu(),
            "image_scores": image_scores.detach().float().cpu()
            if image_scores is not None
            else None,
            "keep_mask": keep_mask.detach().cpu(),
            "keep_indices": [item.detach().cpu() for item in keep_indices],
            "image_embed_shape": list(image_embeds.shape),
            "predictor_features": cast_float_tensor(
                predictor_features, args.feature_dtype
            ),
            "predictor_feature_shape": list(predictor_features.shape),
            "predictor_feature_source": args.feature_source,
            "predictor_early_layer": args.early_layer,
            "vision_score_ms": vision_score_ms,
            "keep_mask_ms": keep_mask_ms,
        }
        if args.save_image_embeds:
            record["image_embeds"] = cast_float_tensor(image_embeds, args.feature_dtype)
        records.append(record)

        if args.log_every > 0 and (
            out_idx == 1 or out_idx == len(indices) or out_idx % args.log_every == 0
        ):
            print(
                f"[TEACHER] {out_idx}/{len(indices)} dataset_index={int(dataset_idx)} "
                f"tokens={int(keep_mask.numel())}->{int(keep_mask.sum().item())} "
                f"vision_score_ms={vision_score_ms:.3f}",
                flush=True,
            )

        if args.shard_size > 0 and len(records) >= args.shard_size:
            shard_dir = output_path
            shard_idx = len(manifest)
            first_dataset_idx = int(records[0]["dataset_index"])
            last_dataset_idx = int(records[-1]["dataset_index"])
            shard_name = (
                f"teacher_scores_shard_{shard_idx:05d}_"
                f"{first_dataset_idx:07d}_{last_dataset_idx:07d}.pt"
            )
            shard_path = shard_dir / shard_name
            save_records(records, shard_path, meta, shard_idx=shard_idx)
            manifest.append({"path": str(shard_path), "records": len(records)})
            records = []

    if args.shard_size > 0:
        if records:
            shard_dir = output_path
            shard_idx = len(manifest)
            first_dataset_idx = int(records[0]["dataset_index"])
            last_dataset_idx = int(records[-1]["dataset_index"])
            shard_name = (
                f"teacher_scores_shard_{shard_idx:05d}_"
                f"{first_dataset_idx:07d}_{last_dataset_idx:07d}.pt"
            )
            shard_path = shard_dir / shard_name
            save_records(records, shard_path, meta, shard_idx=shard_idx)
            manifest.append({"path": str(shard_path), "records": len(records)})
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"meta": meta, "shards": manifest}, indent=2),
            encoding="utf-8",
        )
        print(
            f"[TEACHER] saved manifest with {len(manifest)} shards to {manifest_path}",
            flush=True,
        )
    else:
        save_records(records, output_path, meta)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect topk_attention teacher scores for VisPruner scorer training."
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/wall_x/pretrained/wall-oss-fast",
    )
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/wall_x/datasheet/libero_all",
    )
    parser.add_argument("--repo-id", default="libero_all")
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--feature-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype used when saving predictor_features/image_embeds.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="If >0, save output_path as a directory of sharded .pt files.",
    )
    parser.add_argument(
        "--append-shards",
        action="store_true",
        help="Append new shards after existing shards in output_path.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N samples. Set 0 to suppress per-sample progress.",
    )
    parser.add_argument(
        "--feature-source",
        default="early_hidden",
        choices=["image_embeds", "patch_embeds", "early_hidden"],
        help="Feature type saved as scorer input. early_hidden is the default path toward early pruning.",
    )
    parser.add_argument(
        "--early-layer",
        type=int,
        default=None,
        help="Vision layer index/count used when --feature-source=early_hidden. Defaults to first quarter of layers.",
    )
    parser.add_argument(
        "--output-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/libero_teacher_scores.pt",
    )
    parser.add_argument(
        "--save-image-embeds",
        action="store_true",
        help="Also save final image embeddings as predictor training inputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    collect_records(parse_args())
