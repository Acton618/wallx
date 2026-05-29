import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from wall_x.data.utils import KEY_MAPPINGS
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
)


class IdentityNormalizer:
    def normalize_data(self, xs, dataset_names, *args, **kwargs):
        return xs


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def build_train_config(args, enable_pruning: bool) -> dict:
    model_path = args.model_path
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
        model_cfg = json.load(f)
    strategy = args.pruned_strategy if enable_pruning else "original"
    vispruner_config = {
        "enable": enable_pruning,
        "strategy": strategy,
        "keep_ratio": args.keep_ratio if enable_pruning else 1.0,
        "min_tokens": 1,
        "force_vision_eager": strategy == "topk_attention",
    }
    if enable_pruning and strategy in {"predictor_score", "predictor_early"}:
        vispruner_config["predictor"] = {
            "checkpoint": args.predictor_checkpoint,
            "source": args.predictor_source,
            "early_layer": args.predictor_early_layer,
        }

    return {
        "processor_path": model_path,
        "dof_config": model_cfg["dof_config"],
        "agent_pos_config": model_cfg["agent_pos_config"],
        "data": {
            "use_state_string_representation": False,
            "action_horizon_flow": 32,
        },
        "vispruner": vispruner_config,
    }


def load_model(args, enable_pruning: bool):
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        args.model_path,
        train_config=build_train_config(args, enable_pruning),
    )
    if args.image_min_pixels is not None:
        model.processor.image_processor.min_pixels = int(args.image_min_pixels)
    if args.image_max_pixels is not None:
        model.processor.image_processor.max_pixels = int(args.image_max_pixels)
    if args.image_min_pixels is not None or args.image_max_pixels is not None:
        model.processor.image_processor.size = {
            "shortest_edge": model.processor.image_processor.min_pixels,
            "longest_edge": model.processor.image_processor.max_pixels,
        }
    model.eval()
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        model.to_bfloat16_for_selected_params()
    return model


def format_text(prompt: str, media_kind: str, pred_horizon: int) -> str:
    role_start = "<|im_start|>"
    role_end = "<|im_end|>"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    pad_token = "<|image_pad|>" if media_kind == "image" else "<|video_pad|>"
    propri = "<|propri|>"
    action = "<|action|>"

    return (
        f"{role_start}system\nYou are a helpful assistant.{role_end}\n"
        f"{role_start}user\n"
        f"Observation: front view: {vision_start}{pad_token}{vision_end}\n"
        f"Instruction: {prompt}\n"
        f"Predict the next action in robot action.\n"
        f"Proprioception: {propri}\n"
        f"{role_end}\n"
        f"{role_start}assistant\n"
        f"{action * pred_horizon}{role_end}\n"
    )


def replace_vision_placeholders(processor, text: str, image_grid_thw=None, video_grid_thw=None) -> str:
    merge_length = processor.image_processor.merge_size**2
    if image_grid_thw is not None:
        for grid in image_grid_thw:
            token_count = int(grid.prod().item() // merge_length)
            text = text.replace("<|image_pad|>", "<|placeholder|>" * token_count, 1)
        text = text.replace("<|placeholder|>", "<|image_pad|>")
    if video_grid_thw is not None:
        for grid in video_grid_thw:
            token_count = int(grid.prod().item() // merge_length)
            text = text.replace("<|video_pad|>", "<|placeholder|>" * token_count, 1)
        text = text.replace("<|placeholder|>", "<|video_pad|>")
    return text


def build_common_fields(inputs, processor, args):
    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    inputs["labels"] = None
    inputs["moe_token_types"] = inputs["input_ids"] == action_token_id

    state = torch.zeros((1, 1, args.action_dim), dtype=torch.float32)
    inputs["proprioception"] = state
    inputs["agent_pos_mask"] = torch.ones_like(state)
    inputs["dataset_names"] = ["x2_normal"]

    dof_mask = torch.ones((1, args.pred_horizon, args.action_dim), dtype=torch.float32)
    inputs["dof_mask"] = dof_mask
    return inputs


def make_image_batch(model, sample, image_key: str, args):
    image = sample[image_key]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = (image.clamp(0, 1) * 255).to(torch.uint8).numpy()
    image = Image.fromarray(image).convert("RGB")
    image_inputs = model.processor.image_processor(images=[[image]], return_tensors="pt")
    text = replace_vision_placeholders(
        model.processor,
        format_text(sample.get("task", "pick up the object"), "image", args.pred_horizon),
        image_grid_thw=image_inputs["image_grid_thw"],
    )
    text_inputs = model.processor.tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    inputs = BatchFeature(data={**text_inputs, **image_inputs})
    inputs = build_common_fields(inputs, model.processor, args)
    return inputs.to(args.device)


def sample_video_frames(video_path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def make_video_batch(model, video_path: Path, prompt: str, args):
    frames = sample_video_frames(video_path, args.video_frames)
    video_inputs = model.processor.video_processor(videos=[frames], return_tensors="pt")
    text = replace_vision_placeholders(
        model.processor,
        format_text(prompt, "video", args.pred_horizon),
        video_grid_thw=video_inputs["video_grid_thw"],
    )
    text_inputs = model.processor.tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    inputs = BatchFeature(data={**text_inputs, **video_inputs})
    inputs = build_common_fields(inputs, model.processor, args)
    return inputs.to(args.device)


def count_tokens(model, batch, media_kind: str) -> int:
    token_id = model.config.image_token_id if media_kind == "image" else model.config.video_token_id
    mask = batch["input_ids"] == token_id
    if batch.get("attention_mask") is not None:
        mask = mask & batch["attention_mask"].bool()
    return int(mask.sum().item())


@torch.no_grad()
def resolve_tokens_after(model, batch, media_kind: str) -> int:
    if media_kind != "image" or not model.vispruner.enabled:
        return count_tokens(model, batch, media_kind)

    _, input_ids, attention_mask, _, _, _, pruned = model._encode_images_and_maybe_prune(
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        input_ids=batch.get("input_ids"),
        attention_mask=batch.get("attention_mask"),
        labels=None,
        moe_token_types=batch.get("moe_token_types"),
        position_ids=batch.get("position_ids"),
        video_grid_thw=None,
        second_per_grid_ts=None,
    )
    if not pruned:
        return count_tokens(model, batch, media_kind)
    mask = input_ids == model.config.image_token_id
    if attention_mask is not None:
        mask = mask & attention_mask.bool()
    return int(mask.sum().item())


def average_timing_dict(timing_dicts):
    keys = sorted({key for item in timing_dicts for key in item})
    return {
        key: sum(item.get(key, 0.0) for item in timing_dicts) / len(timing_dicts)
        for key in keys
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def run_records(args, media_kind: str, case_name: str, enable_pruning: bool, items):
    if not items:
        return [], {}

    model = load_model(args, enable_pruning)
    records = []
    timing_counts = {}
    for idx, item in enumerate(items, start=1):
        sync_if_cuda(args.device)
        prepare_start = time.perf_counter()
        if media_kind == "image":
            batch = make_image_batch(model, item["sample"], item["image_key"], args)
        else:
            batch = make_video_batch(model, item["video_path"], item["prompt"], args)
        sync_if_cuda(args.device)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        tokens_before = count_tokens(model, batch, media_kind)
        tokens_after = resolve_tokens_after(model, batch, media_kind)

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
        timing_counts = dict(out.get("timing_counts", timing_counts))
        avg_timing = average_timing_dict(timing_dicts)

        record = {
            "media_kind": media_kind,
            "case": case_name,
            "index": idx,
            "source": item["source"],
            "external_prepare_batch_ms": prepare_ms,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "timings_ms": avg_timing,
        }
        records.append(record)
        print(
            f"[LEROBOT_MEDIA] {media_kind} {case_name} {idx}/{len(items)} "
            f"tokens={tokens_before}->{tokens_after} "
            f"total_ms={avg_timing.get('total_time', 0.0):.3f}",
            flush=True,
        )

    del model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, timing_counts


def summarize(records):
    timing_keys = sorted({key for rec in records for key in rec["timings_ms"]})
    return {
        "num_samples": len(records),
        "tokens_before": mean(rec["tokens_before"] for rec in records),
        "tokens_after": mean(rec["tokens_after"] for rec in records),
        "external_prepare_batch_ms": mean(rec["external_prepare_batch_ms"] for rec in records),
        "timings_ms": {
            key: mean(rec["timings_ms"].get(key, 0.0) for rec in records)
            for key in timing_keys
        },
    }


def paired_deltas(baseline_records, pruned_records):
    rows = []
    for base, pruned in zip(baseline_records, pruned_records):
        base_total = base["timings_ms"].get("total_time", 0.0)
        pruned_total = pruned["timings_ms"].get("total_time", 0.0)
        rows.append(
            {
                "index": base["index"],
                "source": base["source"],
                "tokens_before": base["tokens_before"],
                "tokens_after": pruned["tokens_after"],
                "token_delta": pruned["tokens_after"] - base["tokens_before"],
                "token_delta_pct": (
                    (pruned["tokens_after"] - base["tokens_before"]) / base["tokens_before"] * 100.0
                    if base["tokens_before"]
                    else 0.0
                ),
                "baseline_total_ms": base_total,
                "pruned_total_ms": pruned_total,
                "total_delta_ms": pruned_total - base_total,
                "total_delta_pct": ((pruned_total - base_total) / base_total * 100.0 if base_total else 0.0),
            }
        )
    return rows


def make_items(args):
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    image_key = args.image_key
    if image_key is None:
        image_key = next(iter(KEY_MAPPINGS[args.repo_id]["camera"].keys()))

    image_indices = np.linspace(0, max(0, len(dataset) - 1), args.num_images, dtype=int)
    image_items = []
    for idx in image_indices:
        sample = dataset[int(idx)]
        image_items.append(
            {
                "sample": sample,
                "image_key": image_key,
                "source": f"dataset_index={int(idx)} image_key={image_key}",
            }
        )

    video_dir = Path(args.dataset_root) / "videos" / "chunk-000" / image_key
    video_paths = sorted(video_dir.glob("episode_*.mp4"))[: args.num_videos]
    if len(video_paths) < args.num_videos:
        raise RuntimeError(f"Only found {len(video_paths)} videos under {video_dir}")
    video_items = [
        {
            "video_path": path,
            "prompt": "pick up the object and complete the task",
            "source": str(path),
        }
        for path in video_paths
    ]
    return image_items, video_items


def append_summary(report, title, baseline_summary, pruned_summary):
    report.append(f"## {title}\n")
    base_total = baseline_summary["timings_ms"].get("total_time", 0.0)
    pruned_total = pruned_summary["timings_ms"].get("total_time", 0.0)
    token_delta = pruned_summary["tokens_after"] - baseline_summary["tokens_before"]
    token_delta_pct = (
        token_delta / baseline_summary["tokens_before"] * 100.0
        if baseline_summary["tokens_before"]
        else 0.0
    )
    total_delta = pruned_total - base_total
    total_delta_pct = total_delta / base_total * 100.0 if base_total else 0.0
    report.append(f"- samples: `{baseline_summary['num_samples']}`")
    report.append(
        f"- tokens: baseline `{baseline_summary['tokens_before']:.2f}`, "
        f"pruned `{pruned_summary['tokens_after']:.2f}`, "
        f"delta `{token_delta:+.2f}` (`{token_delta_pct:+.2f}%`)"
    )
    report.append(
        f"- total_time: baseline `{base_total:.3f} ms`, pruned `{pruned_total:.3f} ms`, "
        f"delta `{total_delta:+.3f} ms` (`{total_delta_pct:+.2f}%`)"
    )
    report.append("")

    keys = [
        "total_time",
        "external_prepare_batch_ms",
        "embed_processing",
        "image_path_total",
        "vision_image_forward",
        "vision_image_encode",
        "vision_image_encode_score",
        "vision_video_forward",
        "scatter_video_embeds",
        "vispruner_total",
        "vispruner_build_keep_mask",
        "vispruner_topk_select",
        "vispruner_predictor_score",
        "vision_image_encode_early_prune",
        "vispruner_apply_keep_to_sequences",
        "position_encoding",
        "prefetch_forward",
        "prefill_transformer",
        "cache_preprocessing",
        "ode_integration",
        "ode_transformer_total",
        "postprocessing",
    ]
    all_keys = sorted(set(baseline_summary["timings_ms"]) | set(pruned_summary["timings_ms"]))
    report.append("| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |")
    report.append("|---|---:|---:|---:|---:|")
    seen = set()
    for key in keys + all_keys:
        if key in seen:
            continue
        seen.add(key)
        if key == "external_prepare_batch_ms":
            base = baseline_summary["external_prepare_batch_ms"]
            pruned = pruned_summary["external_prepare_batch_ms"]
        else:
            base = baseline_summary["timings_ms"].get(key, 0.0)
            pruned = pruned_summary["timings_ms"].get(key, 0.0)
        if base == 0.0 and pruned == 0.0:
            continue
        pct = (pruned - base) / base * 100.0 if base else 0.0
        report.append(f"| `{key}` | {base:.3f} | {pruned:.3f} | {pruned-base:+.3f} | {pct:+.2f}% |")
    report.append("")


def write_outputs(args, records, counts):
    image_base = records["image"]["baseline"]
    image_pruned = records["image"]["pruned"]
    video_base = records["video"]["baseline"]
    video_pruned = records["video"]["pruned"]

    summaries = {
        "image": {
            "baseline": summarize(image_base),
            "pruned": summarize(image_pruned),
            "paired_deltas": paired_deltas(image_base, image_pruned),
        },
        "video": {
            "baseline": summarize(video_base),
            "pruned": summarize(video_pruned),
            "paired_deltas": paired_deltas(video_base, video_pruned),
        },
    }

    report = [
        "# Wall-X LeRobot Media VisPruner Timing Report\n",
        f"- dataset_root: `{args.dataset_root}`",
        f"- repo_id: `{args.repo_id}`",
        f"- model_path: `{args.model_path}`",
        f"- num_images: `{args.num_images}`",
        f"- num_videos: `{args.num_videos}`",
        f"- video_frames_per_sample: `{args.video_frames}`",
        f"- warmup: `{args.warmup}`",
        f"- iters: `{args.iters}`",
        f"- keep_ratio: `{args.keep_ratio}`",
        f"- pruned_strategy: `{args.pruned_strategy}`",
        f"- predictor_checkpoint: `{args.predictor_checkpoint}`",
        f"- predictor_source: `{args.predictor_source}`",
        f"- predictor_early_layer: `{args.predictor_early_layer}`",
        f"- image_min_pixels: `{args.image_min_pixels}`",
        f"- image_max_pixels: `{args.image_max_pixels}`",
        f"- device: `{args.device}`",
        "",
        "> Note: current VisPruner hard-pruning is wired to image tokens only. "
        "Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.",
        "",
    ]
    append_summary(report, "Image Samples", summaries["image"]["baseline"], summaries["image"]["pruned"])
    append_summary(report, "Video Samples", summaries["video"]["baseline"], summaries["video"]["pruned"])

    for media_kind in ("image", "video"):
        report.append(f"## {media_kind.title()} Paired Samples\n")
        report.append("| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |")
        report.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in summaries[media_kind]["paired_deltas"]:
            report.append(
                f"| {row['index']} | `{Path(row['source']).name if media_kind == 'video' else row['source']}` | "
                f"{row['tokens_before']} | {row['tokens_after']} | {row['token_delta_pct']:+.2f}% | "
                f"{row['baseline_total_ms']:.3f} | {row['pruned_total_ms']:.3f} | "
                f"{row['total_delta_ms']:+.3f} | {row['total_delta_pct']:+.2f}% |"
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
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument(
        "--pruned-strategy",
        default="topk_attention",
        choices=["topk_attention", "predictor_score", "predictor_early", "norm"],
    )
    parser.add_argument("--predictor-checkpoint", default=None)
    parser.add_argument(
        "--predictor-source",
        default="early_hidden",
        choices=["image_embeds", "patch_embeds", "early_hidden"],
    )
    parser.add_argument("--predictor-early-layer", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--report-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_media_40img_20vid_keep_0.5_report.md",
    )
    parser.add_argument(
        "--results-json",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_media_40img_20vid_keep_0.5_results.json",
    )
    args = parser.parse_args()

    image_items, video_items = make_items(args)
    records = {"image": {}, "video": {}}
    counts = {"image": {}, "video": {}}

    records["image"]["baseline"], counts["image"]["baseline"] = run_records(
        args, "image", "baseline_no_pruning", False, image_items
    )
    records["image"]["pruned"], counts["image"]["pruned"] = run_records(
        args, "image", "vispruner_pruned", True, image_items
    )
    records["video"]["baseline"], counts["video"]["baseline"] = run_records(
        args, "video", "baseline_no_pruning", False, video_items
    )
    records["video"]["pruned"], counts["video"]["pruned"] = run_records(
        args, "video", "vispruner_pruned", True, video_items
    )

    write_outputs(args, records, counts)
    print(f"[LEROBOT_MEDIA] report={args.report_path}", flush=True)
    print(f"[LEROBOT_MEDIA] results_json={args.results_json}", flush=True)


if __name__ == "__main__":
    main()
