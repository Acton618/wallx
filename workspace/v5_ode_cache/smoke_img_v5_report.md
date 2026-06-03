# Wall-X V5 ODE Cache Image Dataset Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `1`
- vispruner_enable: `True`
- keep_ratio: `0.5`
- num_inference_timesteps: `10`
- warmup: `0`
- iters: `1`
- base_seed: `1234`
- device: `cuda`

## V5 Cases

| case | label | early_stop | cache | interval | start_step |
|---|---|---:|---:|---:|---:|
| `fixed_10` | Fixed 10 updates | `False` | `False` | `-` | `-` |
| `cache_i2` | V5 ODE cache interval 2 | `False` | `True` | `2` | `2` |
| `cache_i3` | V5 ODE cache interval 3 | `False` | `True` | `3` | `2` |
| `early_tradeoff` | V3 early stop tradeoff | `True` | `False` | `-` | `-` |

## Summary

| case | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | cache_refreshes | cache_hits | cache_hit_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 380.081 | +0.00% | 291.660 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 219.798 | -42.17% | 159.549 | -45.30% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008979 | 0.011212 | 0.033085 |
| `cache_i3` | 155.927 | -58.98% | 95.933 | -67.11% | 10.00 | 3.00 | 6.00 | 66.67% | 0.014979 | 0.018543 | 0.057783 |
| `early_tradeoff` | 285.458 | -24.90% | 225.565 | -22.66% | 8.00 | 7.00 | 0.00 | 0.00% | 0.520605 | 0.665357 | 1.491592 |

## Interpretation

- `fixed_10` is the exact no-cache baseline with the original 10 Euler updates.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is reported as action difference against `fixed_10` under the same sample and seed.
- Fine-grained timings use `profile_timing=True`, so absolute numbers include CUDA event synchronization overhead; paired deltas are the useful signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 380.081 | 219.798 | 155.927 | 285.458 | -42.17% |
| `external_prepare_batch_ms` | 8.434 | 4.019 | 3.611 | 3.478 | -52.34% |
| `embed_processing` | 34.972 | 29.431 | 29.235 | 29.051 | -15.84% |
| `image_path_total` | 29.874 | 29.013 | 28.855 | 28.664 | -2.88% |
| `vision_image_forward` | 29.904 | 29.040 | 28.881 | 28.691 | -2.89% |
| `position_encoding` | 0.189 | 0.124 | 0.111 | 0.113 | -34.50% |
| `action_initialization` | 6.732 | 0.443 | 0.436 | 0.435 | -93.42% |
| `prefetch_forward` | 40.062 | 28.805 | 28.789 | 28.880 | -28.10% |
| `prefill_transformer` | 39.783 | 28.601 | 28.589 | 28.677 | -28.11% |
| `cache_preprocessing` | 6.306 | 1.310 | 1.294 | 1.280 | -79.22% |
| `ode_integration` | 291.660 | 159.549 | 95.933 | 225.565 | -45.30% |
| `ode_transformer_total` | 285.771 | 156.140 | 93.732 | 219.250 | -45.36% |
| `ode_action_embed_total` | 2.839 | 1.544 | 0.916 | 2.197 | -45.62% |
| `ode_prepare_inputs` | 0.665 | 0.367 | 0.220 | 0.522 | -44.86% |
| `ode_action_head_total` | 0.979 | 0.546 | 0.329 | 0.767 | -44.27% |
| `postprocessing` | 0.007 | 0.006 | 0.006 | 0.006 | -10.91% |
| `action_init_embed` | 4.435 | 0.260 | 0.259 | 0.257 | -94.14% |
| `action_init_noise` | 2.103 | 0.050 | 0.046 | 0.045 | -97.63% |
| `attention_mask_to_device` | 0.008 | 0.007 | 0.006 | 0.006 | -20.62% |
| `embed_tokens` | 0.368 | 0.051 | 0.038 | 0.036 | -86.04% |
| `image_cast` | 0.088 | 0.040 | 0.035 | 0.035 | -54.73% |
| `kv_cache_trim` | 0.863 | 0.854 | 0.860 | 0.850 | -1.12% |
| `moe_indices` | 0.164 | 0.104 | 0.089 | 0.092 | -36.89% |
| `ode_cache_hit` | 0.000 | 0.025 | 0.035 | 0.000 | +0.00% |
| `ode_cache_refresh` | 291.134 | 159.070 | 95.485 | 223.419 | -45.36% |
| `postfix_mask_build` | 0.325 | 0.138 | 0.123 | 0.120 | -57.67% |
| `postfix_moe_indices` | 0.127 | 0.103 | 0.101 | 0.100 | -19.11% |
| `postfix_slice` | 0.057 | 0.057 | 0.057 | 0.057 | -0.78% |
| `prefill_action_head` | 0.221 | 0.148 | 0.145 | 0.148 | -33.07% |
| `prefix_length_resolve` | 4.828 | 0.074 | 0.066 | 0.066 | -98.47% |
| `pruning_position_ids_prepare` | 0.650 | 0.578 | 0.574 | 0.568 | -11.18% |
| `scatter_action_init` | 0.086 | 0.058 | 0.058 | 0.058 | -32.25% |
| `scatter_image_embeds` | 0.228 | 0.111 | 0.102 | 0.101 | -51.57% |
| `scatter_proprioception` | 4.358 | 0.133 | 0.122 | 0.127 | -96.95% |
| `vision_image_encode_score` | 28.070 | 27.432 | 27.288 | 27.081 | -2.27% |
| `vispruner_apply_keep_to_sequences` | 0.211 | 0.197 | 0.196 | 0.195 | -6.83% |
| `vispruner_build_keep_mask` | 0.258 | 0.227 | 0.223 | 0.220 | -11.99% |
| `vispruner_gather_image_embeds` | 0.048 | 0.042 | 0.044 | 0.042 | -12.03% |
| `vispruner_image_lengths` | 0.081 | 0.074 | 0.073 | 0.072 | -9.04% |
| `vispruner_pad_pruned_batch` | 0.077 | 0.072 | 0.072 | 0.073 | -6.67% |
| `vispruner_rope_deltas` | 0.131 | 0.123 | 0.122 | 0.121 | -6.34% |
| `vispruner_score_prepare` | 0.051 | 0.038 | 0.037 | 0.036 | -26.07% |
| `vispruner_topk_select` | 0.135 | 0.120 | 0.118 | 0.116 | -11.25% |
| `vispruner_total` | 0.916 | 0.841 | 0.835 | 0.857 | -8.27% |

## Per-Sample Paired Results

| idx | source | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 380.081 | 219.798 | 155.927 | 285.458 | 0.008979 | 0.014979 | 0.520605 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/v5_ode_cache/smoke_img_v5_results.json`