# Wall-X V5 ODE Cache Video Dataset Report

- video_dir: `/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg`
- video_glob: `episode_*.mp4`
- num_videos: `1`
- video_frames_per_clip: `4`
- prompt: `pick up the object`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- media_type: `video`
- vispruner.prune_video: `True`
- vispruner.keep_ratio: `0.5`
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

| case | video_tokens_before | expected_video_tokens_after | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | cache_refreshes | cache_hits | cache_hit_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 162.00 | 81.00 | 609.766 | +0.00% | 296.812 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 162.00 | 81.00 | 244.291 | -59.94% | 164.498 | -44.58% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008723 | 0.010812 | 0.040594 |
| `cache_i3` | 162.00 | 81.00 | 178.052 | -70.80% | 100.392 | -66.18% | 10.00 | 3.00 | 6.00 | 66.67% | 0.014165 | 0.017733 | 0.057002 |
| `early_tradeoff` | 162.00 | 81.00 | 310.356 | -49.10% | 232.956 | -21.51% | 8.00 | 7.00 | 0.00 | 0.00% | 0.520685 | 0.665823 | 1.480715 |

## Interpretation

- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.
- `expected_video_tokens_after` is the V4 internal video token count after VisPruner. The raw batch still contains the original placeholders before the model prunes them.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is action difference against `fixed_10` under the same video sample and seed.
- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 609.766 | 244.291 | 178.052 | 310.356 | -59.94% |
| `external_prepare_batch_ms` | 27.746 | 15.675 | 13.100 | 15.084 | -43.51% |
| `embed_processing` | 269.512 | 45.228 | 45.466 | 45.287 | -83.22% |
| `vision_video_forward` | 266.994 | 44.830 | 45.080 | 44.893 | -83.21% |
| `scatter_video_embeds` | 0.617 | 0.112 | 0.111 | 0.111 | -81.92% |
| `position_encoding` | 0.201 | 0.123 | 0.119 | 0.117 | -38.70% |
| `action_initialization` | 3.001 | 0.458 | 0.450 | 0.457 | -84.73% |
| `prefetch_forward` | 36.679 | 32.502 | 30.154 | 30.058 | -11.39% |
| `prefill_transformer` | 36.355 | 32.290 | 29.944 | 29.845 | -11.18% |
| `cache_preprocessing` | 3.407 | 1.342 | 1.338 | 1.341 | -60.60% |
| `ode_integration` | 296.812 | 164.498 | 100.392 | 232.956 | -44.58% |
| `ode_transformer_total` | 290.804 | 161.027 | 98.115 | 227.104 | -44.63% |
| `ode_action_embed_total` | 2.873 | 1.548 | 0.938 | 2.210 | -46.12% |
| `ode_prepare_inputs` | 0.686 | 0.376 | 0.228 | 0.538 | -45.14% |
| `ode_action_head_total` | 1.010 | 0.556 | 0.344 | 0.781 | -45.00% |
| `postprocessing` | 0.007 | 0.006 | 0.006 | 0.007 | -9.91% |
| `action_init_embed` | 2.083 | 0.267 | 0.266 | 0.269 | -87.16% |
| `action_init_noise` | 0.705 | 0.054 | 0.049 | 0.049 | -92.34% |
| `attention_mask_to_device` | 0.010 | 0.007 | 0.007 | 0.006 | -30.00% |
| `embed_tokens` | 0.336 | 0.050 | 0.046 | 0.055 | -85.04% |
| `kv_cache_trim` | 0.889 | 0.864 | 0.870 | 0.864 | -2.84% |
| `moe_indices` | 0.176 | 0.100 | 0.098 | 0.095 | -43.13% |
| `ode_cache_hit` | 0.000 | 0.026 | 0.036 | 0.000 | +0.00% |
| `ode_cache_refresh` | 296.263 | 164.002 | 99.928 | 231.329 | -44.64% |
| `postfix_mask_build` | 0.346 | 0.139 | 0.137 | 0.132 | -59.77% |
| `postfix_moe_indices` | 0.131 | 0.111 | 0.107 | 0.107 | -14.82% |
| `postfix_slice` | 0.059 | 0.058 | 0.058 | 0.057 | -0.87% |
| `prefill_action_head` | 0.261 | 0.155 | 0.152 | 0.156 | -40.47% |
| `prefix_length_resolve` | 1.874 | 0.078 | 0.077 | 0.075 | -95.85% |
| `scatter_action_init` | 0.089 | 0.060 | 0.057 | 0.061 | -33.15% |
| `scatter_proprioception` | 1.434 | 0.137 | 0.132 | 0.131 | -90.44% |
| `video_cast` | 0.091 | 0.059 | 0.057 | 0.060 | -35.77% |
| `video_path_total` | 266.961 | 44.799 | 45.050 | 44.865 | -83.22% |
| `video_pruning_position_ids_prepare` | 4.408 | 0.667 | 0.665 | 0.651 | -84.88% |
| `vision_video_encode_score` | 248.231 | 42.995 | 43.266 | 43.107 | -82.68% |
| `vispruner_apply_keep_to_sequences` | 0.308 | 0.209 | 0.210 | 0.206 | -32.23% |
| `vispruner_build_keep_mask` | 10.014 | 0.271 | 0.269 | 0.261 | -97.29% |
| `vispruner_gather_image_embeds` | 0.075 | 0.045 | 0.045 | 0.044 | -39.83% |
| `vispruner_image_lengths` | 3.090 | 0.083 | 0.083 | 0.083 | -97.31% |
| `vispruner_pad_pruned_batch` | 0.117 | 0.082 | 0.078 | 0.079 | -29.94% |
| `vispruner_rope_deltas` | 0.280 | 0.142 | 0.131 | 0.130 | -49.08% |
| `vispruner_score_prepare` | 6.110 | 0.053 | 0.050 | 0.050 | -99.14% |
| `vispruner_topk_select` | 3.768 | 0.144 | 0.147 | 0.138 | -96.17% |
| `vispruner_total` | 14.024 | 0.943 | 0.926 | 0.912 | -93.28% |

## Per-Video Paired Results

| idx | source | video_grid_thw | second_per_grid_ts | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000000.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 609.766 | 244.291 | 178.052 | 310.356 | 0.008723 | 0.014165 | 0.520685 |

## Raw Results
- `/root/autodl-tmp/wall_x/workspace/v5_ode_cache/smoke_video_v5_results.json`