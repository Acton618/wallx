import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import _InferencePerfTimer
from scripts.profile_vispruner_lerobot_media import (
    count_tokens,
    load_model,
    make_image_batch,
    make_items,
    mean,
    sync_if_cuda,
)


FRONT_DIRECT_KEYS = (
    "image_path_total",
    "scatter_image_embeds",
    "position_encoding",
    "prefill_transformer",
)
FRONT_TOP_LEVEL_KEYS = (
    "embed_processing",
    "position_encoding",
    "prefetch_forward",
)
DOWNSTREAM_TOKEN_KEYS = (
    "scatter_image_embeds",
    "position_encoding",
    "prefill_transformer",
)
FLAT_FRONT_CHAIN_STAGES = (
    (
        "s01_image_cast",
        ("image_cast",),
        "Cast pixel_values to the vision tower dtype.",
    ),
    (
        "s02_vision_encode_or_prune",
        (
            "vision_image_encode",
            "vision_image_encode_score",
            "vision_image_encode_early_prune",
        ),
        "Run the active vision path: original encode, attention-score encode, or early-prune encode.",
    ),
    (
        "s03_pruning_position_prepare",
        ("pruning_position_ids_prepare",),
        "Prepare position ids before sequence pruning. Zero for non-pruned baseline.",
    ),
    (
        "s04_apply_pruning",
        ("vispruner_total",),
        "Apply token keep mask to image embeds and sequence tensors. Zero for non-pruned baseline.",
    ),
    (
        "s05_embed_tokens",
        ("embed_tokens",),
        "Embed text/control placeholder token ids.",
    ),
    (
        "s06_scatter_image_embeds",
        ("scatter_image_embeds",),
        "Write image embeddings into image-token positions.",
    ),
    (
        "s07_scatter_proprioception",
        ("scatter_proprioception",),
        "Write proprioception embeddings into proprioception-token positions.",
    ),
    (
        "s08_attention_mask_to_device",
        ("attention_mask_to_device",),
        "Move attention mask to the model device.",
    ),
    (
        "s09_position_and_moe_index",
        ("position_encoding",),
        "Prepare position encoding and MoE token grouping.",
    ),
    (
        "s10_action_initialization",
        ("action_initialization",),
        "Initialize noisy action and write initial action embeddings.",
    ),
    (
        "s11_prefill_transformer",
        ("prefill_transformer",),
        "Run the main transformer prefill.",
    ),
    (
        "s12_prefill_action_head",
        ("prefill_action_head",),
        "Run the first action head projection after prefill.",
    ),
)


@torch.no_grad()
def run_front_chain(model, batch, args):
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    moe_token_types = batch.get("moe_token_types")
    position_ids = batch.get("position_ids")
    proprioception = batch.get("proprioception")
    agent_pos_mask = batch.get("agent_pos_mask")
    dataset_names = batch.get("dataset_names")
    dof_mask = batch.get("dof_mask")

    timer_device = input_ids.device
    perf_timer = _InferencePerfTimer(True, timer_device)
    perf_timer.start("total_time")

    batch_size = input_ids.shape[0]
    prefix_length = None
    start_indices = None
    end_indices = None
    labels = None
    video_grid_thw = None
    second_per_grid_ts = None
    cache_position = None
    past_key_values = None
    positional_masks = None

    perf_timer.start("embed_processing")
    image_embeds = None
    image_pruned = False
    if pixel_values is not None:
        perf_timer.start("vision_image_forward")
        (
            image_embeds,
            input_ids,
            attention_mask,
            labels,
            moe_token_types,
            position_ids,
            image_pruned,
        ) = model._encode_images_and_maybe_prune(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            moe_token_types=moe_token_types,
            position_ids=position_ids,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            perf_timer=perf_timer,
        )
        perf_timer.stop("vision_image_forward")
        if image_pruned:
            start_indices, end_indices = None, None
            prefix_length = None

    perf_timer.start("embed_tokens")
    inputs_embeds = model.model.embed_tokens(input_ids)
    perf_timer.stop("embed_tokens")
    if image_embeds is not None:
        perf_timer.start("scatter_image_embeds")
        inputs_embeds = model._scatter_image_embeds(
            inputs_embeds, input_ids, image_embeds
        )
        perf_timer.stop("scatter_image_embeds")

    if proprioception is not None and not model.config.use_state_string_representation:
        perf_timer.start("scatter_proprioception")
        proprioception = proprioception.to(inputs_embeds.device)
        agent_pos_mask = agent_pos_mask.to(inputs_embeds.device)
        proprio_embed = model.action_preprocessor.proprioception_proj(
            proprioception,
            dataset_names,
            agent_pos_mask,
            use_history=proprioception.shape[1] > 1,
        )
        proprioception_mask = (
            input_ids == model.action_token_id_set["propri_token_id"]
        )
        inputs_embeds[proprioception_mask] = proprio_embed.reshape(
            -1, inputs_embeds.shape[-1]
        ).to(inputs_embeds.dtype)
        perf_timer.stop("scatter_proprioception")

    if attention_mask is not None:
        perf_timer.start("attention_mask_to_device")
        attention_mask = attention_mask.to(inputs_embeds.device)
        perf_timer.stop("attention_mask_to_device")
    perf_timer.stop("embed_processing")

    perf_timer.start("position_encoding")
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if (
            (cache_position is not None and cache_position[0] == 0)
            or model.rope_deltas is None
            or (past_key_values is None)
        ):
            perf_timer.start("position_ids_rope")
            position_ids, rope_deltas = model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            model.rope_deltas = rope_deltas
            perf_timer.stop("position_ids_rope")

    if start_indices is None or end_indices is None:
        perf_timer.start("moe_indices")
        group_size = torch.zeros(
            model.config.num_experts, dtype=torch.long, device="cpu"
        )
        for i in range(model.config.num_experts):
            group_size[i] = (moe_token_types == i).sum()
        start_indices = torch.cumsum(group_size, dim=0) - group_size
        end_indices = torch.cumsum(group_size, dim=0)
        perf_timer.stop("moe_indices")
    perf_timer.stop("position_encoding")

    perf_timer.start("action_initialization")
    perf_timer.start("action_init_noise")
    noise = torch.randn(
        size=(batch_size, args.pred_horizon, args.action_dim),
        dtype=torch.float32,
        device=inputs_embeds.device,
    )
    noisy_action = noise.clone()
    dof_mask = dof_mask.to(inputs_embeds.device).to(torch.float32)
    if args.num_inference_timesteps not in model.times_cache:
        model.times_cache[args.num_inference_timesteps] = torch.linspace(
            0.0,
            1.0,
            args.num_inference_timesteps + 1,
            device=inputs_embeds.device,
            dtype=torch.float32,
        )
    times = model.times_cache[args.num_inference_timesteps]
    time_0 = times[0].unsqueeze(0).repeat(noisy_action.shape[0])
    perf_timer.stop("action_init_noise")

    perf_timer.start("action_init_embed")
    action_embed, adarms_cond = model.action_preprocessor.step(
        timestep=time_0, noisy_action=noisy_action, dof_mask=dof_mask
    )
    action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1]).to(
        inputs_embeds.dtype
    )
    perf_timer.stop("action_init_embed")

    perf_timer.start("scatter_action_init")
    flow_action_mask = input_ids == model.action_token_id_set["action_token_id"]
    inputs_embeds[flow_action_mask] = action_embed
    perf_timer.stop("scatter_action_init")
    perf_timer.stop("action_initialization")

    perf_timer.start("prefetch_forward")
    perf_timer.start("prefill_transformer")
    prefetch_output = model.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        moe_token_types=moe_token_types,
        start_indices=start_indices,
        end_indices=end_indices,
        positional_masks=positional_masks,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        adarms_conds=[None, adarms_cond],
    )
    perf_timer.stop("prefill_transformer")

    perf_timer.start("prefill_action_head")
    action_hidden_states = prefetch_output.last_hidden_state[flow_action_mask].to(
        torch.float32
    )
    _ = model.action_preprocessor.action_proj_back(
        action_hidden_states[:, : model.action_preprocessor.action_hidden_size]
    )
    perf_timer.stop("prefill_action_head")
    perf_timer.stop("prefetch_forward")

    perf_timer.stop("total_time")
    tokens_after = int(
        ((input_ids == model.config.image_token_id) & attention_mask.bool())
        .sum()
        .item()
    )
    return perf_timer.timings_ms, perf_timer.counts, tokens_after


def average_dict(items):
    keys = sorted({key for item in items for key in item})
    return {
        key: sum(item.get(key, 0.0) for item in items) / len(items)
        for key in keys
    }


def summarize(records):
    timings = average_dict([record["timings_ms"] for record in records])
    front_direct = sum(timings.get(key, 0.0) for key in FRONT_DIRECT_KEYS)
    front_top_level = sum(timings.get(key, 0.0) for key in FRONT_TOP_LEVEL_KEYS)
    downstream_token = sum(timings.get(key, 0.0) for key in DOWNSTREAM_TOKEN_KEYS)
    return {
        "num_samples": len(records),
        "tokens_before": mean(record["tokens_before"] for record in records),
        "tokens_after": mean(record["tokens_after"] for record in records),
        "external_prepare_batch_ms": mean(
            record["external_prepare_batch_ms"] for record in records
        ),
        "front_direct_ms": front_direct,
        "front_top_level_ms": front_top_level,
        "downstream_token_ms": downstream_token,
        "timings_ms": timings,
    }


@torch.no_grad()
def run_case(args, case_name, enable_pruning, items):
    model = load_model(args, enable_pruning)
    records = []
    timing_counts = {}

    for warm_idx, item in enumerate(items[: args.warmup_samples], start=1):
        batch = make_image_batch(model, item["sample"], args.image_key, args)
        _ = run_front_chain(model, batch, args)
        sync_if_cuda(args.device)
        del batch
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            f"[FRONT_CHAIN] warmup {case_name} {warm_idx}/{args.warmup_samples}",
            flush=True,
        )

    for idx, item in enumerate(items, start=1):
        sync_if_cuda(args.device)
        prepare_start = time.perf_counter()
        batch = make_image_batch(model, item["sample"], args.image_key, args)
        sync_if_cuda(args.device)
        external_prepare_batch_ms = (time.perf_counter() - prepare_start) * 1000.0

        tokens_before = count_tokens(model, batch, "image")
        timings, counts, tokens_after = run_front_chain(model, batch, args)
        sync_if_cuda(args.device)
        for key, value in counts.items():
            timing_counts[key] = max(timing_counts.get(key, 0), value)
        records.append(
            {
                "case_name": case_name,
                "sample_index": idx,
                "source": item["source"],
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "external_prepare_batch_ms": external_prepare_batch_ms,
                "timings_ms": timings,
            }
        )
        if (
            idx == 1
            or idx == len(items)
            or idx % max(1, args.progress_interval) == 0
        ):
            print(
                f"[FRONT_CHAIN] {case_name} {idx}/{len(items)} "
                f"tokens={tokens_before}->{tokens_after} "
                f"front_direct={sum(timings.get(k, 0.0) for k in FRONT_DIRECT_KEYS):.3f}ms",
                flush=True,
            )
        del batch
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return records, timing_counts


def write_outputs(args, records, timing_counts):
    summaries = {
        "baseline": summarize(records["baseline"]),
        "pruned": summarize(records["pruned"]),
    }
    output = {
        "args": vars(args),
        "records": records,
        "summaries": summaries,
        "timing_counts": timing_counts,
        "front_direct_keys": FRONT_DIRECT_KEYS,
        "front_top_level_keys": FRONT_TOP_LEVEL_KEYS,
        "downstream_token_keys": DOWNSTREAM_TOKEN_KEYS,
        "flat_front_chain_stages": [
            {"name": name, "keys": keys, "description": description}
            for name, keys, description in FLAT_FRONT_CHAIN_STAGES
        ],
    }
    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    def metric_value(summary, metric):
        if metric in summary:
            return summary[metric]
        return summary["timings_ms"].get(metric, 0.0)

    def delta_pct(delta, baseline):
        return delta / baseline * 100.0 if baseline else 0.0

    def direction(delta):
        if delta < -1e-6:
            return "decrease"
        if delta > 1e-6:
            return "increase"
        return "flat"

    def flat_stage_value(summary, keys):
        return sum(summary["timings_ms"].get(key, 0.0) for key in keys)

    def flat_stage_line(order, name, keys, description):
        bv = flat_stage_value(summaries["baseline"], keys)
        pv = flat_stage_value(summaries["pruned"], keys)
        delta = pv - bv
        return (
            f"| {order} | `{name}` | {bv:.3f} | {pv:.3f} | "
            f"{delta:+.3f} | {delta_pct(delta, bv):+.2f}% | "
            f"{direction(delta)} | {description} |"
        )

    b = summaries["baseline"]
    p = summaries["pruned"]
    token_drop = b["tokens_after"] - p["tokens_after"]
    token_drop_pct = token_drop / b["tokens_after"] * 100.0 if b["tokens_after"] else 0.0
    total_delta = metric_value(p, "total_time") - metric_value(b, "total_time")
    lines = [
        "# Wall-X VisPruner Front-Chain Timing Report",
        "",
        f"- dataset_root: `{args.dataset_root}`",
        f"- repo_id: `{args.repo_id}`",
        f"- samples: `{args.num_images}`",
        f"- warmup_samples: `{args.warmup_samples}`",
        f"- keep_ratio: `{args.keep_ratio}`",
        f"- pruned_strategy: `{args.pruned_strategy}`",
        f"- predictor_checkpoint: `{args.predictor_checkpoint}`",
        f"- predictor_source: `{args.predictor_source}`",
        f"- predictor_early_layer: `{args.predictor_early_layer}`",
        f"- image_min_pixels: `{args.image_min_pixels}`",
        f"- image_max_pixels: `{args.image_max_pixels}`",
        f"- device: `{args.device}`",
        "",
        "## Summary",
        "",
        f"- tokens: baseline `{b['tokens_after']:.2f}`, pruned `{p['tokens_after']:.2f}`",
        f"- token_drop: `{token_drop:.2f}` (`{token_drop_pct:.2f}%`)",
        f"- front_chain_total_time: baseline `{metric_value(b, 'total_time'):.3f}`, pruned `{metric_value(p, 'total_time'):.3f}`, delta `{total_delta:+.3f}`",
        "",
        "## Sequential Front-Chain Stages",
        "",
        "These rows describe the front-chain execution as a flat pipeline. "
        "`front_chain_total_time` is the final total timestamp for this front-chain profiler.",
        "",
        "| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction | meaning |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    flat_baseline_sum = 0.0
    flat_pruned_sum = 0.0
    for order, (name, keys, description) in enumerate(FLAT_FRONT_CHAIN_STAGES, start=1):
        flat_baseline_sum += flat_stage_value(summaries["baseline"], keys)
        flat_pruned_sum += flat_stage_value(summaries["pruned"], keys)
        lines.append(flat_stage_line(order, name, keys, description))
    baseline_unattributed = metric_value(b, "total_time") - flat_baseline_sum
    pruned_unattributed = metric_value(p, "total_time") - flat_pruned_sum
    unattributed_delta = pruned_unattributed - baseline_unattributed
    lines.extend(
        [
            f"| 13 | `s13_unattributed_framework_overhead` | {baseline_unattributed:.3f} | {pruned_unattributed:.3f} | {unattributed_delta:+.3f} | {delta_pct(unattributed_delta, baseline_unattributed):+.2f}% | {direction(unattributed_delta)} | Small untracked gaps between explicit stages; kept so the pipeline sums to total_time. |",
            f"| total | `front_chain_total_time` | {metric_value(b, 'total_time'):.3f} | {metric_value(p, 'total_time'):.3f} | {total_delta:+.3f} | {delta_pct(total_delta, metric_value(b, 'total_time')):+.2f}% | {direction(total_delta)} | Final total timestamp for front-chain profiling. |",
            "",
            "## Timestamp Definitions",
            "",
            "| order | timestamp | meaning in the front token-processing chain |",
            "|---:|---|---|",
        ]
    )
    for order, (name, _, description) in enumerate(FLAT_FRONT_CHAIN_STAGES, start=1):
        lines.append(f"| {order} | `{name}` | {description} |")
    lines.extend(
        [
            "| 13 | `s13_unattributed_framework_overhead` | Small untracked gaps between explicit stages; kept so the pipeline sums to `front_chain_total_time`. |",
            "| total | `front_chain_total_time` | Final total timestamp from entering the model front chain to finishing `prefill_action_head`. |",
            "",
        ]
    )

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
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument(
        "--pruned-strategy",
        default="predictor_early",
        choices=["topk_attention", "predictor_score", "predictor_early", "norm"],
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
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--results-json", required=True)
    args = parser.parse_args()

    image_items, _ = make_items(args)
    records = {}
    counts = {}
    records["baseline"], counts["baseline"] = run_case(
        args, "baseline_no_pruning", False, image_items
    )
    records["pruned"], counts["pruned"] = run_case(
        args, "vispruner_pruned", True, image_items
    )
    write_outputs(args, records, counts)
    print(f"[FRONT_CHAIN] report={args.report_path}", flush=True)
    print(f"[FRONT_CHAIN] results_json={args.results_json}", flush=True)


if __name__ == "__main__":
    main()
