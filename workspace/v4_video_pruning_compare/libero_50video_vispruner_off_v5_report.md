# Wall-X V5 ODE Cache Video Dataset Report

- video_dir: `/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg`
- video_glob: `episode_*.mp4`
- num_videos: `50`
- video_frames_per_clip: `4`
- prompt: `pick up the object`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- media_type: `video`
- vispruner.prune_video: `False`
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
| `fixed_10` | 162.00 | 162.00 | 374.109 | +0.00% | 293.070 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 162.00 | 162.00 | 239.632 | -35.95% | 163.204 | -44.31% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008398 | 0.010472 | 0.034591 |
| `cache_i3` | 162.00 | 162.00 | 172.093 | -54.00% | 96.504 | -67.07% | 10.00 | 3.00 | 6.00 | 66.67% | 0.014056 | 0.017583 | 0.054988 |
| `early_tradeoff` | 162.00 | 162.00 | 304.843 | -18.51% | 228.409 | -22.06% | 8.00 | 7.00 | 0.00 | 0.00% | 0.519180 | 0.663187 | 1.480248 |

## Interpretation

- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.
- `expected_video_tokens_after` is the V4 internal video token count after VisPruner. The raw batch still contains the original placeholders before the model prunes them.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is action difference against `fixed_10` under the same video sample and seed.
- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 374.109 | 239.632 | 172.093 | 304.843 | -35.95% |
| `external_prepare_batch_ms` | 13.443 | 13.306 | 13.279 | 13.403 | -1.02% |
| `embed_processing` | 47.837 | 43.804 | 43.463 | 43.783 | -8.43% |
| `vision_video_forward` | 47.300 | 43.379 | 43.058 | 43.359 | -8.29% |
| `scatter_video_embeds` | 0.198 | 0.135 | 0.128 | 0.136 | -32.08% |
| `position_encoding` | 0.822 | 0.786 | 0.739 | 0.746 | -4.29% |
| `action_initialization` | 0.516 | 0.468 | 0.455 | 0.464 | -9.34% |
| `prefetch_forward` | 30.344 | 29.902 | 29.485 | 29.966 | -1.46% |
| `prefill_transformer` | 30.139 | 29.699 | 29.285 | 29.763 | -1.46% |
| `cache_preprocessing` | 1.381 | 1.329 | 1.311 | 1.334 | -3.74% |
| `ode_integration` | 293.070 | 163.204 | 96.504 | 228.409 | -44.31% |
| `ode_transformer_total` | 287.163 | 159.722 | 94.265 | 223.465 | -44.38% |
| `ode_action_embed_total` | 2.794 | 1.559 | 0.933 | 2.183 | -44.20% |
| `ode_prepare_inputs` | 0.680 | 0.378 | 0.224 | 0.530 | -44.42% |
| `ode_action_head_total` | 1.007 | 0.562 | 0.333 | 0.791 | -44.21% |
| `postprocessing` | 0.007 | 0.007 | 0.006 | 0.007 | -0.13% |
| `action_init_embed` | 0.311 | 0.277 | 0.271 | 0.274 | -10.91% |
| `action_init_noise` | 0.065 | 0.051 | 0.047 | 0.051 | -21.15% |
| `attention_mask_to_device` | 0.007 | 0.007 | 0.007 | 0.007 | +5.33% |
| `embed_tokens` | 0.055 | 0.048 | 0.043 | 0.048 | -12.43% |
| `kv_cache_trim` | 0.862 | 0.862 | 0.861 | 0.864 | -0.02% |
| `moe_indices` | 0.101 | 0.098 | 0.094 | 0.098 | -2.38% |
| `ode_cache_hit` | 0.000 | 0.025 | 0.036 | 0.000 | +0.00% |
| `ode_cache_refresh` | 292.539 | 162.714 | 96.048 | 227.673 | -44.38% |
| `position_ids_rope` | 0.674 | 0.641 | 0.600 | 0.602 | -4.97% |
| `postfix_mask_build` | 0.141 | 0.134 | 0.126 | 0.135 | -4.42% |
| `postfix_moe_indices` | 0.112 | 0.107 | 0.104 | 0.108 | -4.89% |
| `postfix_slice` | 0.058 | 0.058 | 0.057 | 0.058 | +0.41% |
| `prefill_action_head` | 0.155 | 0.153 | 0.149 | 0.153 | -1.24% |
| `prefix_length_resolve` | 0.116 | 0.077 | 0.071 | 0.078 | -33.61% |
| `scatter_action_init` | 0.063 | 0.063 | 0.061 | 0.063 | +0.03% |
| `scatter_proprioception` | 0.180 | 0.137 | 0.132 | 0.137 | -23.46% |
| `video_cast` | 0.060 | 0.057 | 0.056 | 0.058 | -4.43% |
| `video_path_total` | 47.272 | 43.352 | 43.030 | 43.331 | -8.29% |
| `vision_video_encode` | 47.156 | 43.240 | 42.916 | 43.217 | -8.30% |

## Per-Video Paired Results

| idx | source | video_grid_thw | second_per_grid_ts | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000000.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 588.643 | 235.162 | 171.300 | 307.681 | 0.008576 | 0.014236 | 0.520673 |
| 2 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000020.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 374.065 | 244.523 | 171.134 | 301.298 | 0.008850 | 0.014316 | 0.522086 |
| 3 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000040.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.364 | 244.704 | 172.960 | 306.935 | 0.008254 | 0.013756 | 0.520019 |
| 4 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000061.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 375.733 | 243.005 | 173.135 | 305.943 | 0.008554 | 0.014627 | 0.521396 |
| 5 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000081.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.388 | 242.766 | 172.241 | 302.376 | 0.008327 | 0.013815 | 0.519440 |
| 6 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000101.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.134 | 243.354 | 171.055 | 303.364 | 0.008136 | 0.014247 | 0.518639 |
| 7 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000122.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.018 | 235.136 | 171.465 | 303.711 | 0.008201 | 0.013821 | 0.519702 |
| 8 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000142.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.105 | 235.730 | 172.305 | 307.743 | 0.008430 | 0.013990 | 0.521299 |
| 9 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000163.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.219 | 242.034 | 171.743 | 299.822 | 0.008505 | 0.014598 | 0.518313 |
| 10 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000183.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 380.129 | 240.997 | 171.406 | 301.185 | 0.008034 | 0.013324 | 0.517245 |
| 11 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000203.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.576 | 241.690 | 172.313 | 303.688 | 0.007804 | 0.013224 | 0.518147 |
| 12 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000224.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.152 | 241.406 | 171.193 | 305.120 | 0.008458 | 0.014350 | 0.519711 |
| 13 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000244.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 376.749 | 241.810 | 171.311 | 301.272 | 0.008383 | 0.013857 | 0.519975 |
| 14 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000265.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 377.762 | 245.629 | 171.354 | 302.546 | 0.008297 | 0.014102 | 0.517170 |
| 15 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000285.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.279 | 244.746 | 173.925 | 301.506 | 0.008607 | 0.013274 | 0.519285 |
| 16 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000305.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.974 | 246.139 | 172.830 | 302.179 | 0.008519 | 0.013470 | 0.517239 |
| 17 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000326.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.188 | 244.563 | 171.617 | 312.930 | 0.008533 | 0.013589 | 0.517968 |
| 18 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000346.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.426 | 243.046 | 173.298 | 302.817 | 0.008530 | 0.014233 | 0.520010 |
| 19 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000366.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 378.090 | 245.014 | 170.852 | 301.615 | 0.008662 | 0.014091 | 0.517923 |
| 20 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000387.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.670 | 238.941 | 171.292 | 304.422 | 0.008324 | 0.014561 | 0.520444 |
| 21 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000407.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 377.197 | 237.239 | 171.164 | 302.661 | 0.008610 | 0.013785 | 0.520975 |
| 22 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000428.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 374.774 | 236.059 | 171.262 | 300.606 | 0.008412 | 0.013946 | 0.518379 |
| 23 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000448.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.208 | 233.807 | 172.695 | 301.471 | 0.008689 | 0.013594 | 0.520216 |
| 24 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000468.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.304 | 234.219 | 172.690 | 300.531 | 0.008457 | 0.014345 | 0.517989 |
| 25 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000489.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.089 | 241.808 | 173.312 | 300.204 | 0.008546 | 0.014651 | 0.517898 |
| 26 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000509.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.064 | 239.692 | 175.196 | 300.063 | 0.008519 | 0.013805 | 0.519117 |
| 27 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000530.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 376.369 | 239.414 | 172.585 | 309.095 | 0.008725 | 0.014406 | 0.520580 |
| 28 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000550.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 378.339 | 242.628 | 172.343 | 306.123 | 0.008473 | 0.015039 | 0.520807 |
| 29 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000570.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.013 | 239.431 | 171.480 | 305.741 | 0.008229 | 0.013847 | 0.518995 |
| 30 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000591.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.558 | 240.551 | 171.732 | 301.049 | 0.008133 | 0.013402 | 0.517631 |
| 31 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000611.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.374 | 238.308 | 171.746 | 300.657 | 0.008572 | 0.014244 | 0.517341 |
| 32 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000632.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.862 | 243.557 | 173.313 | 303.951 | 0.007992 | 0.014315 | 0.518036 |
| 33 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000652.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.018 | 236.030 | 172.350 | 310.067 | 0.008493 | 0.014396 | 0.520622 |
| 34 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000672.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.529 | 234.842 | 171.861 | 307.344 | 0.007962 | 0.013910 | 0.520326 |
| 35 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000693.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.016 | 235.073 | 170.460 | 304.794 | 0.008388 | 0.013799 | 0.518980 |
| 36 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000713.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.185 | 243.766 | 171.068 | 306.424 | 0.008358 | 0.013812 | 0.519189 |
| 37 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000733.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 360.550 | 250.118 | 171.452 | 302.311 | 0.008188 | 0.014197 | 0.519679 |
| 38 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000754.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.827 | 241.929 | 171.165 | 299.543 | 0.008162 | 0.014162 | 0.518344 |
| 39 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000774.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 377.270 | 234.819 | 170.550 | 309.160 | 0.008160 | 0.014043 | 0.517293 |
| 40 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000795.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.463 | 235.095 | 173.280 | 305.959 | 0.008117 | 0.013299 | 0.517842 |
| 41 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000815.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 374.198 | 236.632 | 171.715 | 305.205 | 0.008104 | 0.014134 | 0.519648 |
| 42 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000835.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 377.618 | 236.325 | 172.669 | 306.863 | 0.008010 | 0.014568 | 0.520599 |
| 43 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000856.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.798 | 237.281 | 171.787 | 309.337 | 0.008367 | 0.013876 | 0.518591 |
| 44 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000876.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.727 | 234.956 | 172.432 | 307.485 | 0.008653 | 0.014664 | 0.517848 |
| 45 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000897.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.159 | 234.944 | 171.685 | 310.904 | 0.008375 | 0.014517 | 0.521379 |
| 46 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000917.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.837 | 235.508 | 171.048 | 311.630 | 0.008732 | 0.014719 | 0.518456 |
| 47 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000937.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 362.777 | 236.648 | 175.390 | 307.336 | 0.008721 | 0.014422 | 0.518246 |
| 48 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000958.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 362.553 | 235.442 | 171.455 | 309.413 | 0.008591 | 0.013238 | 0.518598 |
| 49 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000978.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.027 | 236.719 | 172.289 | 310.110 | 0.008262 | 0.014123 | 0.518091 |
| 50 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000999.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.106 | 238.356 | 173.765 | 307.973 | 0.008912 | 0.014076 | 0.520611 |

## Raw Results
- `/root/autodl-tmp/wall_x/workspace/v4_video_pruning_compare/libero_50video_vispruner_off_v5_results.json`