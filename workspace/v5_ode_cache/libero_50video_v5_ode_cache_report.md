# Wall-X V5 ODE Cache Video Dataset Report

- video_dir: `/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg`
- video_glob: `episode_*.mp4`
- num_videos: `50`
- video_frames_per_clip: `4`
- prompt: `pick up the object`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- media_type: `video`
- vispruner.prune_video: `True`
- vispruner.keep_ratio: `0.5`
- num_inference_timesteps: `10`
- warmup: `1`
- iters: `2`
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
| `fixed_10` | 162.00 | 81.00 | 369.372 | +0.00% | 291.709 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 162.00 | 81.00 | 238.762 | -35.36% | 162.192 | -44.40% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008320 | 0.010447 | 0.035004 |
| `cache_i3` | 162.00 | 81.00 | 176.839 | -52.12% | 99.170 | -66.00% | 10.00 | 3.00 | 6.00 | 66.67% | 0.013822 | 0.017336 | 0.054817 |
| `early_tradeoff` | 162.00 | 81.00 | 306.465 | -17.03% | 228.815 | -21.56% | 8.00 | 7.00 | 0.00 | 0.00% | 0.519293 | 0.664076 | 1.479059 |

## Interpretation

- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.
- `expected_video_tokens_after` is the V4 internal video token count after VisPruner. The raw batch still contains the original placeholders before the model prunes them.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is action difference against `fixed_10` under the same video sample and seed.
- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 369.372 | 238.762 | 176.839 | 306.465 | -35.36% |
| `external_prepare_batch_ms` | 13.948 | 13.216 | 12.856 | 13.366 | -5.24% |
| `embed_processing` | 45.559 | 45.101 | 45.417 | 45.510 | -1.01% |
| `vision_video_forward` | 45.172 | 44.734 | 45.039 | 45.127 | -0.97% |
| `scatter_video_embeds` | 0.112 | 0.105 | 0.108 | 0.110 | -6.24% |
| `position_encoding` | 0.123 | 0.115 | 0.119 | 0.121 | -6.69% |
| `action_initialization` | 0.458 | 0.442 | 0.462 | 0.458 | -3.56% |
| `prefetch_forward` | 30.035 | 29.458 | 30.208 | 30.086 | -1.92% |
| `prefill_transformer` | 29.823 | 29.251 | 29.995 | 29.874 | -1.92% |
| `cache_preprocessing` | 1.351 | 1.320 | 1.330 | 1.340 | -2.28% |
| `ode_integration` | 291.709 | 162.192 | 99.170 | 228.815 | -44.40% |
| `ode_transformer_total` | 285.807 | 158.745 | 96.866 | 223.890 | -44.46% |
| `ode_action_embed_total` | 2.786 | 1.536 | 0.960 | 2.180 | -44.88% |
| `ode_prepare_inputs` | 0.670 | 0.374 | 0.230 | 0.530 | -44.16% |
| `ode_action_head_total` | 1.016 | 0.558 | 0.349 | 0.799 | -45.08% |
| `postprocessing` | 0.007 | 0.006 | 0.006 | 0.007 | -3.88% |
| `action_init_embed` | 0.270 | 0.262 | 0.276 | 0.271 | -3.18% |
| `action_init_noise` | 0.050 | 0.046 | 0.048 | 0.050 | -6.82% |
| `attention_mask_to_device` | 0.007 | 0.007 | 0.007 | 0.007 | -0.11% |
| `embed_tokens` | 0.046 | 0.041 | 0.043 | 0.046 | -12.44% |
| `kv_cache_trim` | 0.873 | 0.861 | 0.862 | 0.862 | -1.36% |
| `moe_indices` | 0.100 | 0.093 | 0.097 | 0.099 | -7.13% |
| `ode_cache_hit` | 0.000 | 0.025 | 0.036 | 0.000 | +0.00% |
| `ode_cache_refresh` | 291.176 | 161.703 | 98.706 | 228.102 | -44.47% |
| `postfix_mask_build` | 0.142 | 0.132 | 0.137 | 0.143 | -7.17% |
| `postfix_moe_indices` | 0.109 | 0.107 | 0.110 | 0.110 | -2.34% |
| `postfix_slice` | 0.057 | 0.059 | 0.058 | 0.058 | +3.38% |
| `prefill_action_head` | 0.154 | 0.151 | 0.156 | 0.154 | -1.98% |
| `prefix_length_resolve` | 0.079 | 0.071 | 0.075 | 0.079 | -10.88% |
| `scatter_action_init` | 0.061 | 0.059 | 0.061 | 0.060 | -3.28% |
| `scatter_proprioception` | 0.131 | 0.126 | 0.131 | 0.131 | -3.75% |
| `video_cast` | 0.052 | 0.046 | 0.048 | 0.052 | -11.23% |
| `video_path_total` | 45.143 | 44.706 | 45.011 | 45.098 | -0.97% |
| `video_pruning_position_ids_prepare` | 0.675 | 0.635 | 0.671 | 0.680 | -6.03% |
| `vision_video_encode_score` | 43.351 | 43.009 | 43.247 | 43.307 | -0.79% |
| `vispruner_apply_keep_to_sequences` | 0.207 | 0.203 | 0.205 | 0.207 | -2.00% |
| `vispruner_build_keep_mask` | 0.267 | 0.249 | 0.260 | 0.267 | -6.63% |
| `vispruner_gather_image_embeds` | 0.045 | 0.044 | 0.045 | 0.044 | -1.92% |
| `vispruner_image_lengths` | 0.084 | 0.079 | 0.082 | 0.083 | -6.52% |
| `vispruner_pad_pruned_batch` | 0.080 | 0.076 | 0.078 | 0.080 | -4.20% |
| `vispruner_rope_deltas` | 0.132 | 0.127 | 0.132 | 0.132 | -3.81% |
| `vispruner_score_prepare` | 0.049 | 0.044 | 0.047 | 0.049 | -10.17% |
| `vispruner_topk_select` | 0.146 | 0.134 | 0.141 | 0.146 | -7.59% |
| `vispruner_total` | 0.926 | 0.886 | 0.910 | 0.921 | -4.31% |

## Per-Video Paired Results

| idx | source | video_grid_thw | second_per_grid_ts | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000000.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.052 | 235.697 | 175.125 | 305.469 | 0.008024 | 0.013708 | 0.521175 |
| 2 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000020.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.304 | 235.999 | 173.199 | 304.834 | 0.009260 | 0.013840 | 0.521367 |
| 3 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000040.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.506 | 236.355 | 174.996 | 305.650 | 0.008334 | 0.014739 | 0.520142 |
| 4 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000061.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.909 | 235.817 | 174.446 | 306.879 | 0.008082 | 0.014628 | 0.521287 |
| 5 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000081.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.431 | 235.830 | 173.662 | 308.258 | 0.008083 | 0.013203 | 0.520217 |
| 6 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000101.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.991 | 235.387 | 175.654 | 306.998 | 0.008479 | 0.013100 | 0.520152 |
| 7 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000122.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.339 | 235.664 | 179.103 | 307.644 | 0.008144 | 0.013478 | 0.518836 |
| 8 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000142.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.288 | 236.552 | 175.500 | 303.823 | 0.008325 | 0.013403 | 0.520413 |
| 9 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000163.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.764 | 236.394 | 175.015 | 308.661 | 0.008134 | 0.013986 | 0.519216 |
| 10 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000183.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 380.373 | 237.381 | 174.147 | 312.342 | 0.007922 | 0.013250 | 0.517739 |
| 11 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000203.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.158 | 235.101 | 178.216 | 299.316 | 0.008313 | 0.013562 | 0.518430 |
| 12 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000224.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.619 | 237.188 | 176.345 | 298.841 | 0.008431 | 0.013427 | 0.521772 |
| 13 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000244.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.371 | 235.898 | 178.068 | 300.144 | 0.008518 | 0.014527 | 0.519638 |
| 14 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000265.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.323 | 237.412 | 176.180 | 296.879 | 0.008642 | 0.014332 | 0.518752 |
| 15 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000285.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.087 | 237.634 | 175.352 | 297.918 | 0.008375 | 0.013644 | 0.517492 |
| 16 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000305.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.814 | 237.170 | 174.625 | 301.990 | 0.008336 | 0.013987 | 0.518348 |
| 17 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000326.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.593 | 236.000 | 177.123 | 300.677 | 0.008200 | 0.013697 | 0.518122 |
| 18 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000346.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.728 | 237.899 | 174.779 | 300.028 | 0.008733 | 0.013895 | 0.518746 |
| 19 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000366.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.819 | 237.593 | 173.702 | 299.021 | 0.008460 | 0.014814 | 0.518686 |
| 20 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000387.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.656 | 238.953 | 174.476 | 316.623 | 0.008140 | 0.013851 | 0.520470 |
| 21 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000407.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.673 | 235.749 | 175.564 | 307.445 | 0.008294 | 0.013505 | 0.520890 |
| 22 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000428.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.040 | 236.999 | 177.562 | 302.405 | 0.008488 | 0.013409 | 0.518446 |
| 23 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000448.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.123 | 268.160 | 179.608 | 302.755 | 0.008397 | 0.014281 | 0.519314 |
| 24 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000468.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.900 | 245.223 | 179.953 | 302.303 | 0.008433 | 0.013781 | 0.518688 |
| 25 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000489.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.235 | 265.076 | 178.114 | 302.397 | 0.008092 | 0.013666 | 0.517524 |
| 26 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000509.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.184 | 236.812 | 181.190 | 305.474 | 0.007981 | 0.013940 | 0.518309 |
| 27 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000530.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.298 | 235.614 | 173.932 | 303.245 | 0.008591 | 0.014566 | 0.519254 |
| 28 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000550.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.176 | 235.580 | 180.393 | 303.358 | 0.008267 | 0.014162 | 0.520223 |
| 29 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000570.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.542 | 240.483 | 178.209 | 299.610 | 0.008479 | 0.013498 | 0.518173 |
| 30 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000591.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 375.648 | 236.470 | 178.470 | 301.401 | 0.008114 | 0.013313 | 0.517494 |
| 31 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000611.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.322 | 237.236 | 179.948 | 300.556 | 0.008032 | 0.013698 | 0.519598 |
| 32 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000632.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.156 | 236.764 | 181.077 | 304.428 | 0.008296 | 0.013986 | 0.517180 |
| 33 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000652.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.615 | 235.331 | 177.062 | 323.639 | 0.008106 | 0.013979 | 0.521550 |
| 34 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000672.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.928 | 236.206 | 176.931 | 338.220 | 0.008235 | 0.013658 | 0.520058 |
| 35 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000693.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.052 | 237.997 | 175.395 | 335.508 | 0.008872 | 0.014349 | 0.518758 |
| 36 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000713.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 376.808 | 235.559 | 176.599 | 307.971 | 0.008474 | 0.013429 | 0.518995 |
| 37 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000733.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.578 | 236.026 | 176.949 | 307.383 | 0.008322 | 0.014101 | 0.519683 |
| 38 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000754.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.162 | 236.430 | 190.761 | 308.607 | 0.007928 | 0.014224 | 0.517516 |
| 39 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000774.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.004 | 237.158 | 177.918 | 308.283 | 0.008400 | 0.013888 | 0.517744 |
| 40 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000795.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.109 | 237.513 | 178.497 | 308.794 | 0.008237 | 0.013721 | 0.517808 |
| 41 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000815.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 374.783 | 237.977 | 177.112 | 306.076 | 0.008463 | 0.013631 | 0.521088 |
| 42 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000835.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.252 | 240.883 | 175.364 | 306.434 | 0.008146 | 0.013710 | 0.521363 |
| 43 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000856.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.630 | 238.521 | 176.183 | 303.702 | 0.008008 | 0.013243 | 0.517712 |
| 44 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000876.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.162 | 243.607 | 174.031 | 310.038 | 0.009133 | 0.014177 | 0.519918 |
| 45 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000897.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.256 | 239.159 | 174.892 | 308.314 | 0.007594 | 0.013935 | 0.521579 |
| 46 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000917.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.900 | 241.169 | 176.925 | 308.046 | 0.007852 | 0.013666 | 0.517691 |
| 47 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000937.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.481 | 242.940 | 176.740 | 307.302 | 0.008349 | 0.013472 | 0.519719 |
| 48 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000958.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.369 | 240.610 | 176.018 | 304.289 | 0.008503 | 0.013790 | 0.519731 |
| 49 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000978.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.455 | 239.592 | 175.300 | 305.299 | 0.008393 | 0.013544 | 0.517930 |
| 50 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000999.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.648 | 239.310 | 175.557 | 307.992 | 0.008606 | 0.013717 | 0.519709 |

## Raw Results
- `/root/autodl-tmp/wall_x/workspace/v5_ode_cache/libero_50video_v5_ode_cache_results.json`