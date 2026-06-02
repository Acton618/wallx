# Wall-X V3 ODE Early Stop Video Dataset Report

- video_dir: `/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg`
- video_glob: `episode_*.mp4`
- num_videos: `50`
- video_frames_per_clip: `4`
- prompt: `pick up the object`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- media_type: `video`
- num_inference_timesteps: `10`
- warmup: `1`
- iters: `2`
- base_seed: `1234`
- device: `cuda`

## V3 Cases

| case | enable | threshold | min_steps | patience | metric |
|---|---:|---:|---:|---:|---|
| `fixed_10` | `False` | `-` | `-` | `-` | `mean_abs` |
| `early_safe` | `True` | `0.2` | `2` | `1` | `mean_abs` |
| `early_tradeoff` | `True` | `0.3` | `8` | `1` | `mean_abs` |

## Summary

| case | video_tokens | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | postfix_steps | stopped_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 162.00 | 362.713 | +0.00% | 287.645 | +0.00% | 10.00 | 9.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `early_safe` | 162.00 | 360.721 | -0.55% | 286.249 | -0.49% | 10.00 | 9.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `early_tradeoff` | 162.00 | 296.564 | -18.24% | 222.011 | -22.82% | 8.00 | 7.00 | 100.00% | 0.519164 | 0.663148 | 1.479967 |

## Interpretation

- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.
- Accuracy is action difference against `fixed_10` under the same video sample and seed; it measures how much V3 early stop changes the original fixed-step inference output.
- `actual_updates` counts the existing prefetch update plus later postfix ODE updates. `postfix_steps` is `actual_updates - 1`.
- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.

## Stage Timing

| stage | fixed_10_ms | early_safe_ms | early_tradeoff_ms | tradeoff_delta_vs_fixed |
|---|---:|---:|---:|---:|
| `total_time` | 362.713 | 360.721 | 296.564 | -18.24% |
| `external_prepare_batch_ms` | 13.342 | 13.156 | 13.025 | -2.38% |
| `embed_processing` | 42.854 | 42.742 | 42.811 | -0.10% |
| `vision_video_forward` | 42.379 | 42.294 | 42.358 | -0.05% |
| `scatter_video_embeds` | 0.061 | 0.055 | 0.057 | -6.29% |
| `position_encoding` | 0.762 | 0.729 | 0.738 | -3.25% |
| `action_initialization` | 0.471 | 0.446 | 0.447 | -5.20% |
| `prefetch_forward` | 29.500 | 29.134 | 29.121 | -1.28% |
| `prefill_transformer` | 29.299 | 28.938 | 28.925 | -1.27% |
| `cache_preprocessing` | 1.338 | 1.284 | 1.300 | -2.86% |
| `ode_integration` | 287.645 | 286.249 | 222.011 | -22.82% |
| `ode_transformer_total` | 281.017 | 280.318 | 217.448 | -22.62% |
| `ode_action_embed_total` | 2.736 | 2.730 | 2.119 | -22.54% |
| `ode_prepare_inputs` | 0.659 | 0.654 | 0.507 | -23.16% |
| `ode_action_head_total` | 1.007 | 0.984 | 0.759 | -24.62% |
| `postprocessing` | 0.007 | 0.006 | 0.006 | -3.10% |
| `action_init_embed` | 0.276 | 0.265 | 0.264 | -4.30% |
| `action_init_noise` | 0.051 | 0.046 | 0.048 | -5.15% |
| `attention_mask_to_device` | 0.007 | 0.007 | 0.007 | -8.67% |
| `embed_tokens` | 0.051 | 0.046 | 0.049 | -4.46% |
| `kv_cache_trim` | 0.871 | 0.841 | 0.848 | -2.61% |
| `moe_indices` | 0.098 | 0.092 | 0.092 | -5.29% |
| `position_ids_rope` | 0.617 | 0.592 | 0.600 | -2.80% |
| `postfix_mask_build` | 0.136 | 0.128 | 0.131 | -3.63% |
| `postfix_moe_indices` | 0.107 | 0.104 | 0.104 | -2.74% |
| `postfix_slice` | 0.059 | 0.056 | 0.056 | -4.29% |
| `prefill_action_head` | 0.150 | 0.147 | 0.146 | -2.21% |
| `prefix_length_resolve` | 0.074 | 0.068 | 0.073 | -1.82% |
| `scatter_action_init` | 0.063 | 0.059 | 0.059 | -6.90% |
| `scatter_proprioception` | 0.138 | 0.131 | 0.132 | -4.57% |

## Per-Video Paired Results

| idx | source | video_grid_thw | second_per_grid_ts | safe_updates | tradeoff_updates | fixed_total_ms | safe_total_ms | tradeoff_total_ms | safe_mae | tradeoff_mae |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000000.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 366.566 | 360.154 | 302.156 | 0.000000 | 0.521247 |
| 2 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000020.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 367.908 | 361.748 | 297.151 | 0.000000 | 0.521599 |
| 3 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000040.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 366.707 | 360.593 | 297.663 | 0.000000 | 0.519991 |
| 4 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000061.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 364.827 | 360.763 | 295.052 | 0.000000 | 0.521099 |
| 5 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000081.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 361.606 | 359.186 | 296.350 | 0.000000 | 0.519962 |
| 6 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000101.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 361.585 | 367.643 | 298.468 | 0.000000 | 0.518672 |
| 7 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000122.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.463 | 360.679 | 293.877 | 0.000000 | 0.519662 |
| 8 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000142.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 359.326 | 363.457 | 293.718 | 0.000000 | 0.520656 |
| 9 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000163.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 363.255 | 360.831 | 296.846 | 0.000000 | 0.518814 |
| 10 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000183.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 365.441 | 362.347 | 298.517 | 0.000000 | 0.516918 |
| 11 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000203.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 365.213 | 365.059 | 302.168 | 0.000000 | 0.517700 |
| 12 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000224.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 363.972 | 367.126 | 298.524 | 0.000000 | 0.520740 |
| 13 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000244.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.555 | 366.133 | 301.045 | 0.000000 | 0.519968 |
| 14 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000265.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 364.697 | 364.602 | 296.380 | 0.000000 | 0.518117 |
| 15 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000285.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.660 | 361.295 | 297.211 | 0.000000 | 0.519343 |
| 16 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000305.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 363.028 | 361.442 | 299.434 | 0.000000 | 0.517255 |
| 17 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000326.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.484 | 361.908 | 295.872 | 0.000000 | 0.517790 |
| 18 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000346.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 365.675 | 361.213 | 298.000 | 0.000000 | 0.520170 |
| 19 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000366.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.657 | 355.555 | 298.723 | 0.000000 | 0.517873 |
| 20 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000387.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 364.296 | 354.505 | 296.852 | 0.000000 | 0.519797 |
| 21 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000407.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.217 | 356.245 | 295.744 | 0.000000 | 0.519933 |
| 22 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000428.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 365.333 | 352.950 | 293.171 | 0.000000 | 0.517746 |
| 23 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000448.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.803 | 354.971 | 293.220 | 0.000000 | 0.519785 |
| 24 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000468.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 363.741 | 354.567 | 294.616 | 0.000000 | 0.519036 |
| 25 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000489.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 363.703 | 354.786 | 297.105 | 0.000000 | 0.517690 |
| 26 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000509.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 364.910 | 367.232 | 298.142 | 0.000000 | 0.518351 |
| 27 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000530.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.122 | 364.804 | 296.166 | 0.000000 | 0.520861 |
| 28 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000550.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 361.193 | 357.359 | 296.150 | 0.000000 | 0.519802 |
| 29 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000570.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 358.776 | 359.948 | 298.312 | 0.000000 | 0.518748 |
| 30 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000591.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 361.356 | 358.887 | 293.110 | 0.000000 | 0.518101 |
| 31 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000611.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 362.356 | 363.291 | 293.237 | 0.000000 | 0.517148 |
| 32 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000632.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 359.693 | 360.371 | 292.145 | 0.000000 | 0.517832 |
| 33 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000652.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 361.774 | 360.114 | 291.972 | 0.000000 | 0.520968 |
| 34 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000672.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 359.543 | 360.393 | 293.384 | 0.000000 | 0.520641 |
| 35 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000693.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 360.235 | 354.753 | 296.249 | 0.000000 | 0.517872 |
| 36 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000713.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 357.572 | 355.618 | 293.286 | 0.000000 | 0.519129 |
| 37 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000733.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 357.938 | 355.357 | 294.041 | 0.000000 | 0.520031 |
| 38 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000754.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 358.132 | 361.315 | 295.205 | 0.000000 | 0.517601 |
| 39 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000774.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 357.259 | 360.114 | 298.435 | 0.000000 | 0.517944 |
| 40 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000795.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 358.130 | 360.666 | 296.114 | 0.000000 | 0.518121 |
| 41 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000815.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 357.957 | 361.870 | 297.736 | 0.000000 | 0.519888 |
| 42 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000835.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 358.127 | 362.526 | 296.941 | 0.000000 | 0.520872 |
| 43 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000856.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 358.116 | 365.738 | 297.118 | 0.000000 | 0.518222 |
| 44 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000876.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 361.135 | 362.458 | 297.317 | 0.000000 | 0.518347 |
| 45 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000897.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 380.701 | 366.785 | 296.863 | 0.000000 | 0.521678 |
| 46 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000917.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 366.472 | 364.089 | 297.784 | 0.000000 | 0.518566 |
| 47 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000937.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 366.875 | 361.445 | 297.992 | 0.000000 | 0.518570 |
| 48 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000958.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 368.492 | 360.941 | 297.410 | 0.000000 | 0.519116 |
| 49 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000978.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 363.128 | 359.488 | 296.569 | 0.000000 | 0.517598 |
| 50 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000999.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 358.931 | 360.740 | 298.637 | 0.000000 | 0.520615 |

## Raw Results
- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50video_v3_ode_early_stop_results.json`