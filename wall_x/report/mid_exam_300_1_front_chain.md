# Wall-X VisPruner Front-Chain Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- samples: `300`
- warmup_samples: `5`
- keep_ratio: `0.5`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- image_min_pixels: `254016`
- image_max_pixels: `None`
- device: `cuda`

## Summary

- tokens: baseline `324.00`, pruned `162.00`
- token_drop: `162.00` (`50.00%`)
- front_chain_total_time: baseline `132.071`, pruned `98.886`, delta `-33.186`

## Sequential Front-Chain Stages

These rows describe the front-chain execution as a flat pipeline. `front_chain_total_time` is the final total timestamp for this front-chain profiler.

| order | stage | baseline_ms | pruned_ms | delta_ms | delta_pct | direction | meaning |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `s01_image_cast` | 0.046 | 0.046 | +0.000 | +0.39% | increase | Cast pixel_values to the vision tower dtype. |
| 2 | `s02_vision_encode_or_prune` | 98.171 | 64.365 | -33.807 | -34.44% | decrease | Run the active vision path: original encode, attention-score encode, or early-prune encode. |
| 3 | `s03_pruning_position_prepare` | 0.000 | 0.713 | +0.713 | +0.00% | increase | Prepare position ids before sequence pruning. Zero for non-pruned baseline. |
| 4 | `s04_apply_pruning` | 0.000 | 0.690 | +0.690 | +0.00% | increase | Apply token keep mask to image embeds and sequence tensors. Zero for non-pruned baseline. |
| 5 | `s05_embed_tokens` | 0.049 | 0.045 | -0.004 | -7.65% | decrease | Embed text/control placeholder token ids. |
| 6 | `s06_scatter_image_embeds` | 0.139 | 0.109 | -0.029 | -21.07% | decrease | Write image embeddings into image-token positions. |
| 7 | `s07_scatter_proprioception` | 0.146 | 0.133 | -0.013 | -8.88% | decrease | Write proprioception embeddings into proprioception-token positions. |
| 8 | `s08_attention_mask_to_device` | 0.007 | 0.007 | -0.000 | -2.10% | decrease | Move attention mask to the model device. |
| 9 | `s09_position_and_moe_index` | 0.822 | 0.119 | -0.703 | -85.54% | decrease | Prepare position encoding and MoE token grouping. |
| 10 | `s10_action_initialization` | 0.482 | 0.469 | -0.013 | -2.71% | decrease | Initialize noisy action and write initial action embeddings. |
| 11 | `s11_prefill_transformer` | 31.795 | 31.693 | -0.103 | -0.32% | decrease | Run the main transformer prefill. |
| 12 | `s12_prefill_action_head` | 0.113 | 0.114 | +0.001 | +1.09% | increase | Run the first action head projection after prefill. |
| 13 | `s13_unattributed_framework_overhead` | 0.301 | 0.383 | +0.081 | +27.06% | increase | Small untracked gaps between explicit stages; kept so the pipeline sums to total_time. |
| total | `front_chain_total_time` | 132.071 | 98.886 | -33.186 | -25.13% | decrease | Final total timestamp for front-chain profiling. |

## Timestamp Definitions

| order | timestamp | meaning in the front token-processing chain |
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

