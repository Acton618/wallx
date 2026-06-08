# Wall-X VisPruner Keep Ratio Sweep Report

## Experiment Setup

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- samples: `300`
- image_min_pixels: `254016`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- keep_ratios: `0.7, 0.6, 0.5, 0.4, 0.3`
- baseline visual tokens: `324.00`

## Overall Summary

| keep_ratio | tokens | front_chain baseline | front_chain pruned | delta_ms | delta_pct | s02_delta_ms | pruning_overhead_s03+s04 | MAE | RMSE | cosine | allclose_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 324->227 | 131.191 | 707.515 | +576.324 | +439.30% | +522.831 | 22.155 | 0.018815 | 0.023514 | 0.999977 | 0.0000 |
| 0.6 | 324->195 | 131.191 | 681.509 | +550.318 | +419.48% | +495.779 | 21.944 | 0.021970 | 0.027487 | 0.999968 | 0.0000 |
| 0.5 | 324->162 | 131.191 | 180.719 | +49.528 | +37.75% | +38.668 | 4.089 | 0.025359 | 0.031705 | 0.999958 | 0.0000 |
| 0.4 | 324->130 | 131.191 | 96.263 | -34.928 | -26.62% | -37.863 | 1.380 | 0.028883 | 0.036149 | 0.999945 | 0.0000 |
| 0.3 | 324->98 | 131.191 | 90.506 | -40.685 | -31.01% | -38.710 | 1.402 | 0.033086 | 0.041473 | 0.999928 | 0.0000 |

## Timestamp Definitions

| order | timestamp | meaning |
|---:|---|---|
| 1 | `s01_image_cast` | Cast pixel_values to the vision tower dtype. |
| 2 | `s02_vision_encode_or_prune` | Run the active vision path: original encode, attention-score encode, or early-prune encode. |
| 3 | `s03_pruning_position_prepare` | Prepare position ids before sequence pruning. Zero for non-pruned baseline. |
| 4 | `s04_apply_pruning` | Apply token keep mask to image embeds and sequence tensors. Zero for non-pruned baseline. |
| 5 | `s05_embed_tokens` | Embed text/control placeholder token ids. |
| 6 | `s06_scatter_image_embeds` | Write image embeddings into image-token positions. |
| 7 | `s07_scatter_proprioception` | Write proprioception embeddings into proprioception-token positions. |
| 8 | `s08_attention_mask_to_device` | Move attention mask to the model device. |
| 9 | `s09_position_and_moe_index` | Prepare position encoding and MoE token grouping. |
| 10 | `s10_action_initialization` | Initialize noisy action and write initial action embeddings. |
| 11 | `s11_prefill_transformer` | Run the main transformer prefill. |
| 12 | `s12_prefill_action_head` | Run the first action head projection after prefill. |
| 13 | `s13_unattributed_framework_overhead` | Small untracked gaps between explicit stages; kept so the pipeline sums to `front_chain_total_time`. |
| total | `front_chain_total_time` | Final total timestamp from entering the model front chain to finishing `prefill_action_head`. |

## keep_ratio = 0.7

### Front-Chain 13-Stage Timing

| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 0.050 | 1.480 | +1.429 | +2846.31% | increase |
| 2 | `s02_vision_encode_or_prune` | 96.771 | 619.602 | +522.831 | +540.28% | increase |
| 3 | `s03_pruning_position_prepare` | 0.000 | 10.826 | +10.826 | +0.00% | increase |
| 4 | `s04_apply_pruning` | 0.000 | 11.329 | +11.329 | +0.00% | increase |
| 5 | `s05_embed_tokens` | 0.056 | 0.016 | -0.040 | -72.12% | decrease |
| 6 | `s06_scatter_image_embeds` | 0.152 | 0.767 | +0.614 | +402.69% | increase |
| 7 | `s07_scatter_proprioception` | 0.157 | 0.804 | +0.646 | +410.41% | increase |
| 8 | `s08_attention_mask_to_device` | 0.007 | 0.003 | -0.005 | -64.32% | decrease |
| 9 | `s09_position_and_moe_index` | 0.890 | 1.288 | +0.398 | +44.66% | increase |
| 10 | `s10_action_initialization` | 0.502 | 3.658 | +3.155 | +628.01% | increase |
| 11 | `s11_prefill_transformer` | 32.168 | 41.232 | +9.065 | +28.18% | increase |
| 12 | `s12_prefill_action_head` | 0.130 | 1.564 | +1.434 | +1102.97% | increase |
| 13 | `s13_unattributed_framework_overhead` | 0.306 | 14.948 | +14.641 | +4777.87% | increase |
| total | `front_chain_total_time` | 131.191 | 707.515 | +576.324 | +439.30% | increase |

### Action Output Consistency

- MAE: `0.018815`
- RMSE: `0.023514`
- mean max_abs: `0.075728`
- worst max_abs: `0.149651`
- cosine_similarity: `0.999977`
- allclose_rate: `0.0000`

## keep_ratio = 0.6

### Front-Chain 13-Stage Timing

| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 0.050 | 1.328 | +1.278 | +2544.50% | increase |
| 2 | `s02_vision_encode_or_prune` | 96.771 | 592.550 | +495.779 | +512.32% | increase |
| 3 | `s03_pruning_position_prepare` | 0.000 | 9.801 | +9.801 | +0.00% | increase |
| 4 | `s04_apply_pruning` | 0.000 | 12.143 | +12.143 | +0.00% | increase |
| 5 | `s05_embed_tokens` | 0.056 | 0.020 | -0.036 | -64.12% | decrease |
| 6 | `s06_scatter_image_embeds` | 0.152 | 0.860 | +0.707 | +463.86% | increase |
| 7 | `s07_scatter_proprioception` | 0.157 | 0.852 | +0.695 | +441.23% | increase |
| 8 | `s08_attention_mask_to_device` | 0.007 | 0.003 | -0.004 | -61.25% | decrease |
| 9 | `s09_position_and_moe_index` | 0.890 | 1.580 | +0.690 | +77.49% | increase |
| 10 | `s10_action_initialization` | 0.502 | 4.396 | +3.894 | +775.01% | increase |
| 11 | `s11_prefill_transformer` | 32.168 | 40.678 | +8.510 | +26.46% | increase |
| 12 | `s12_prefill_action_head` | 0.130 | 1.669 | +1.539 | +1183.58% | increase |
| 13 | `s13_unattributed_framework_overhead` | 0.306 | 15.630 | +15.323 | +5000.39% | increase |
| total | `front_chain_total_time` | 131.191 | 681.509 | +550.318 | +419.48% | increase |

### Action Output Consistency

- MAE: `0.021970`
- RMSE: `0.027487`
- mean max_abs: `0.088868`
- worst max_abs: `0.227585`
- cosine_similarity: `0.999968`
- allclose_rate: `0.0000`

## keep_ratio = 0.5

### Front-Chain 13-Stage Timing

| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 0.050 | 0.218 | +0.168 | +334.74% | increase |
| 2 | `s02_vision_encode_or_prune` | 96.771 | 135.439 | +38.668 | +39.96% | increase |
| 3 | `s03_pruning_position_prepare` | 0.000 | 1.907 | +1.907 | +0.00% | increase |
| 4 | `s04_apply_pruning` | 0.000 | 2.182 | +2.182 | +0.00% | increase |
| 5 | `s05_embed_tokens` | 0.056 | 0.044 | -0.012 | -21.28% | decrease |
| 6 | `s06_scatter_image_embeds` | 0.152 | 0.217 | +0.065 | +42.34% | increase |
| 7 | `s07_scatter_proprioception` | 0.157 | 0.243 | +0.086 | +54.30% | increase |
| 8 | `s08_attention_mask_to_device` | 0.007 | 0.006 | -0.001 | -16.31% | decrease |
| 9 | `s09_position_and_moe_index` | 0.890 | 0.308 | -0.582 | -65.36% | decrease |
| 10 | `s10_action_initialization` | 0.502 | 0.947 | +0.445 | +88.55% | increase |
| 11 | `s11_prefill_transformer` | 32.168 | 36.360 | +4.192 | +13.03% | increase |
| 12 | `s12_prefill_action_head` | 0.130 | 0.320 | +0.190 | +146.33% | increase |
| 13 | `s13_unattributed_framework_overhead` | 0.306 | 2.527 | +2.221 | +724.67% | increase |
| total | `front_chain_total_time` | 131.191 | 180.719 | +49.528 | +37.75% | increase |

### Action Output Consistency

- MAE: `0.025359`
- RMSE: `0.031705`
- mean max_abs: `0.100451`
- worst max_abs: `0.203187`
- cosine_similarity: `0.999958`
- allclose_rate: `0.0000`

## keep_ratio = 0.4

### Front-Chain 13-Stage Timing

| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 0.050 | 0.048 | -0.002 | -4.72% | decrease |
| 2 | `s02_vision_encode_or_prune` | 96.771 | 58.908 | -37.863 | -39.13% | decrease |
| 3 | `s03_pruning_position_prepare` | 0.000 | 0.720 | +0.720 | +0.00% | increase |
| 4 | `s04_apply_pruning` | 0.000 | 0.660 | +0.660 | +0.00% | increase |
| 5 | `s05_embed_tokens` | 0.056 | 0.046 | -0.010 | -18.12% | decrease |
| 6 | `s06_scatter_image_embeds` | 0.152 | 0.105 | -0.047 | -31.10% | decrease |
| 7 | `s07_scatter_proprioception` | 0.157 | 0.131 | -0.026 | -16.82% | decrease |
| 8 | `s08_attention_mask_to_device` | 0.007 | 0.006 | -0.001 | -10.66% | decrease |
| 9 | `s09_position_and_moe_index` | 0.890 | 0.118 | -0.772 | -86.71% | decrease |
| 10 | `s10_action_initialization` | 0.502 | 0.481 | -0.022 | -4.33% | decrease |
| 11 | `s11_prefill_transformer` | 32.168 | 34.538 | +2.370 | +7.37% | increase |
| 12 | `s12_prefill_action_head` | 0.130 | 0.121 | -0.009 | -7.14% | decrease |
| 13 | `s13_unattributed_framework_overhead` | 0.306 | 0.382 | +0.076 | +24.71% | increase |
| total | `front_chain_total_time` | 131.191 | 96.263 | -34.928 | -26.62% | decrease |

### Action Output Consistency

- MAE: `0.028883`
- RMSE: `0.036149`
- mean max_abs: `0.113577`
- worst max_abs: `0.239101`
- cosine_similarity: `0.999945`
- allclose_rate: `0.0000`

## keep_ratio = 0.3

### Front-Chain 13-Stage Timing

| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 0.050 | 0.048 | -0.002 | -3.76% | decrease |
| 2 | `s02_vision_encode_or_prune` | 96.771 | 58.061 | -38.710 | -40.00% | decrease |
| 3 | `s03_pruning_position_prepare` | 0.000 | 0.744 | +0.744 | +0.00% | increase |
| 4 | `s04_apply_pruning` | 0.000 | 0.658 | +0.658 | +0.00% | increase |
| 5 | `s05_embed_tokens` | 0.056 | 0.048 | -0.008 | -14.55% | decrease |
| 6 | `s06_scatter_image_embeds` | 0.152 | 0.104 | -0.048 | -31.54% | decrease |
| 7 | `s07_scatter_proprioception` | 0.157 | 0.131 | -0.026 | -16.52% | decrease |
| 8 | `s08_attention_mask_to_device` | 0.007 | 0.006 | -0.001 | -10.80% | decrease |
| 9 | `s09_position_and_moe_index` | 0.890 | 0.118 | -0.772 | -86.72% | decrease |
| 10 | `s10_action_initialization` | 0.502 | 0.474 | -0.028 | -5.64% | decrease |
| 11 | `s11_prefill_transformer` | 32.168 | 29.607 | -2.561 | -7.96% | decrease |
| 12 | `s12_prefill_action_head` | 0.130 | 0.119 | -0.011 | -8.47% | decrease |
| 13 | `s13_unattributed_framework_overhead` | 0.306 | 0.387 | +0.080 | +26.15% | increase |
| total | `front_chain_total_time` | 131.191 | 90.506 | -40.685 | -31.01% | decrease |

### Action Output Consistency

- MAE: `0.033086`
- RMSE: `0.041473`
- mean max_abs: `0.128427`
- worst max_abs: `0.256607`
- cosine_similarity: `0.999928`
- allclose_rate: `0.0000`

## Reading Notes

- The acceleration metric is `front_chain_total_time`, not full ODE/action inference latency.
- `s02_vision_encode_or_prune` is the main place where `predictor_early` should save time.
- `s03_pruning_position_prepare` and `s04_apply_pruning` are expected extra pruning overhead.
- Action metrics compare predictor output against the same-sample original Wall-X output.
