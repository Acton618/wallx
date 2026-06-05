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
| `fixed_10` | 162.00 | 81.00 | 370.337 | +0.00% | 288.863 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 162.00 | 81.00 | 238.488 | -35.60% | 161.361 | -44.14% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008477 | 0.010583 | 0.034346 |
| `cache_i3` | 162.00 | 81.00 | 173.047 | -53.27% | 96.547 | -66.58% | 10.00 | 3.00 | 6.00 | 66.67% | 0.014004 | 0.017500 | 0.053875 |
| `early_tradeoff` | 162.00 | 81.00 | 301.719 | -18.53% | 225.120 | -22.07% | 8.00 | 7.00 | 0.00 | 0.00% | 0.519379 | 0.664165 | 1.480584 |

## Interpretation

- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.
- `expected_video_tokens_after` is the V4 internal video token count after VisPruner. The raw batch still contains the original placeholders before the model prunes them.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is action difference against `fixed_10` under the same video sample and seed.
- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 370.337 | 238.488 | 173.047 | 301.719 | -35.60% |
| `external_prepare_batch_ms` | 13.716 | 13.148 | 13.640 | 13.160 | -4.14% |
| `embed_processing` | 49.783 | 45.551 | 44.985 | 45.063 | -8.50% |
| `vision_video_forward` | 49.358 | 45.153 | 44.611 | 44.686 | -8.52% |
| `scatter_video_embeds` | 0.121 | 0.117 | 0.108 | 0.108 | -3.26% |
| `position_encoding` | 0.121 | 0.120 | 0.114 | 0.116 | -0.97% |
| `action_initialization` | 0.502 | 0.461 | 0.450 | 0.449 | -8.19% |
| `prefetch_forward` | 29.573 | 29.550 | 29.505 | 29.513 | -0.08% |
| `prefill_transformer` | 29.367 | 29.345 | 29.300 | 29.308 | -0.07% |
| `cache_preprocessing` | 1.356 | 1.310 | 1.301 | 1.321 | -3.39% |
| `ode_integration` | 288.863 | 161.361 | 96.547 | 225.120 | -44.14% |
| `ode_transformer_total` | 283.001 | 157.916 | 94.339 | 220.305 | -44.20% |
| `ode_action_embed_total` | 2.749 | 1.540 | 0.921 | 2.132 | -43.98% |
| `ode_prepare_inputs` | 0.670 | 0.377 | 0.224 | 0.525 | -43.69% |
| `ode_action_head_total` | 1.042 | 0.558 | 0.329 | 0.769 | -46.47% |
| `postprocessing` | 0.007 | 0.007 | 0.006 | 0.007 | -2.75% |
| `action_init_embed` | 0.303 | 0.273 | 0.268 | 0.267 | -10.01% |
| `action_init_noise` | 0.062 | 0.051 | 0.047 | 0.048 | -17.74% |
| `attention_mask_to_device` | 0.007 | 0.007 | 0.007 | 0.006 | +0.87% |
| `embed_tokens` | 0.053 | 0.046 | 0.043 | 0.044 | -12.18% |
| `kv_cache_trim` | 0.855 | 0.855 | 0.855 | 0.864 | -0.02% |
| `moe_indices` | 0.099 | 0.097 | 0.092 | 0.095 | -2.28% |
| `ode_cache_hit` | 0.000 | 0.024 | 0.035 | 0.000 | +0.00% |
| `ode_cache_refresh` | 288.340 | 160.878 | 96.099 | 224.412 | -44.21% |
| `postfix_mask_build` | 0.139 | 0.131 | 0.128 | 0.131 | -5.73% |
| `postfix_moe_indices` | 0.105 | 0.105 | 0.103 | 0.105 | -0.13% |
| `postfix_slice` | 0.057 | 0.057 | 0.057 | 0.058 | +0.28% |
| `prefill_action_head` | 0.150 | 0.148 | 0.150 | 0.149 | -1.10% |
| `prefix_length_resolve` | 0.111 | 0.074 | 0.072 | 0.074 | -32.96% |
| `scatter_action_init` | 0.060 | 0.060 | 0.059 | 0.059 | -0.41% |
| `scatter_proprioception` | 0.154 | 0.134 | 0.127 | 0.128 | -13.27% |
| `video_cast` | 0.060 | 0.056 | 0.058 | 0.058 | -6.73% |
| `video_path_total` | 49.328 | 45.123 | 44.581 | 44.657 | -8.53% |
| `video_pruning_position_ids_prepare` | 0.733 | 0.656 | 0.627 | 0.626 | -10.46% |
| `vision_video_encode_score` | 47.208 | 43.344 | 42.868 | 42.943 | -8.18% |
| `vispruner_apply_keep_to_sequences` | 0.207 | 0.209 | 0.203 | 0.202 | +1.13% |
| `vispruner_build_keep_mask` | 0.464 | 0.265 | 0.253 | 0.254 | -42.95% |
| `vispruner_gather_image_embeds` | 0.046 | 0.048 | 0.044 | 0.044 | +3.27% |
| `vispruner_image_lengths` | 0.146 | 0.081 | 0.079 | 0.079 | -44.22% |
| `vispruner_pad_pruned_batch` | 0.079 | 0.079 | 0.077 | 0.078 | -0.22% |
| `vispruner_rope_deltas` | 0.132 | 0.133 | 0.128 | 0.128 | +1.02% |
| `vispruner_score_prepare` | 0.170 | 0.048 | 0.046 | 0.047 | -71.43% |
| `vispruner_topk_select` | 0.219 | 0.143 | 0.136 | 0.137 | -34.63% |
| `vispruner_total` | 1.184 | 0.927 | 0.891 | 0.892 | -21.71% |

## Per-Video Paired Results

| idx | source | video_grid_thw | second_per_grid_ts | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000000.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 608.981 | 236.592 | 172.067 | 297.329 | 0.008723 | 0.014165 | 0.520685 |
| 2 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000020.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 376.652 | 235.826 | 170.583 | 296.652 | 0.008917 | 0.014124 | 0.521552 |
| 3 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000040.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.371 | 239.603 | 174.948 | 296.154 | 0.008648 | 0.014547 | 0.519915 |
| 4 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000061.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.714 | 239.764 | 178.407 | 297.331 | 0.008479 | 0.014486 | 0.521756 |
| 5 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000081.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.730 | 241.784 | 178.359 | 297.069 | 0.008120 | 0.013734 | 0.519608 |
| 6 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000101.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 359.831 | 240.630 | 174.753 | 295.549 | 0.008672 | 0.014373 | 0.519888 |
| 7 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000122.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 360.125 | 236.812 | 173.640 | 296.352 | 0.008213 | 0.013749 | 0.519429 |
| 8 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000142.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 358.575 | 234.003 | 175.036 | 297.344 | 0.008502 | 0.014193 | 0.520671 |
| 9 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000163.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 357.941 | 239.437 | 173.474 | 307.858 | 0.008107 | 0.014194 | 0.518740 |
| 10 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000183.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 358.007 | 241.313 | 170.396 | 306.793 | 0.008351 | 0.013336 | 0.518240 |
| 11 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000203.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 373.844 | 236.386 | 170.021 | 308.839 | 0.008501 | 0.013716 | 0.519134 |
| 12 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000224.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.495 | 239.780 | 170.442 | 296.362 | 0.008235 | 0.013996 | 0.521244 |
| 13 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000244.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 359.670 | 237.415 | 174.661 | 295.853 | 0.008550 | 0.014036 | 0.519704 |
| 14 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000265.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.251 | 237.426 | 170.042 | 296.835 | 0.007998 | 0.013763 | 0.518090 |
| 15 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000285.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.316 | 238.924 | 170.661 | 296.351 | 0.007935 | 0.014620 | 0.517223 |
| 16 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000305.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 359.058 | 238.316 | 173.844 | 295.422 | 0.008533 | 0.013650 | 0.518320 |
| 17 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000326.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 360.318 | 237.279 | 174.411 | 298.047 | 0.008340 | 0.013595 | 0.518157 |
| 18 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000346.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.934 | 239.432 | 174.747 | 295.905 | 0.008666 | 0.013915 | 0.518352 |
| 19 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000366.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 371.282 | 235.841 | 174.447 | 308.955 | 0.008490 | 0.013277 | 0.518909 |
| 20 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000387.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.030 | 239.229 | 170.295 | 308.571 | 0.008569 | 0.014284 | 0.521373 |
| 21 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000407.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.546 | 239.355 | 174.210 | 306.861 | 0.008263 | 0.013284 | 0.521883 |
| 22 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000428.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.532 | 239.519 | 178.790 | 308.614 | 0.008083 | 0.013785 | 0.519016 |
| 23 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000448.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 368.577 | 241.071 | 176.202 | 297.522 | 0.008660 | 0.013837 | 0.519228 |
| 24 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000468.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.642 | 239.329 | 178.875 | 308.308 | 0.008315 | 0.013847 | 0.517922 |
| 25 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000489.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.504 | 233.016 | 174.777 | 306.532 | 0.009205 | 0.014659 | 0.517721 |
| 26 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000509.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.852 | 235.172 | 175.042 | 306.832 | 0.008384 | 0.013108 | 0.519216 |
| 27 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000530.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 362.686 | 237.479 | 170.783 | 305.562 | 0.008426 | 0.014539 | 0.518862 |
| 28 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000550.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 362.494 | 236.815 | 170.834 | 305.353 | 0.008686 | 0.014457 | 0.521296 |
| 29 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000570.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 362.932 | 237.381 | 170.080 | 307.841 | 0.008426 | 0.013468 | 0.518670 |
| 30 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000591.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 362.082 | 237.565 | 174.982 | 308.532 | 0.008304 | 0.013929 | 0.517424 |
| 31 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000611.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.090 | 240.540 | 171.090 | 306.415 | 0.008776 | 0.013685 | 0.519492 |
| 32 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000632.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.941 | 240.613 | 174.068 | 304.557 | 0.008407 | 0.013821 | 0.517527 |
| 33 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000652.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 369.947 | 240.635 | 170.728 | 304.181 | 0.008999 | 0.014399 | 0.521294 |
| 34 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000672.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.402 | 237.162 | 174.674 | 302.302 | 0.008118 | 0.013412 | 0.520066 |
| 35 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000693.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 370.195 | 237.817 | 174.517 | 303.925 | 0.008746 | 0.014930 | 0.519946 |
| 36 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000713.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.359 | 238.921 | 171.906 | 303.710 | 0.008576 | 0.013572 | 0.519034 |
| 37 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000733.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.459 | 240.350 | 173.577 | 301.523 | 0.008351 | 0.014473 | 0.519216 |
| 38 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000754.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 375.908 | 238.989 | 172.958 | 302.626 | 0.008527 | 0.014335 | 0.517870 |
| 39 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000774.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 367.801 | 234.462 | 173.704 | 302.381 | 0.008221 | 0.013567 | 0.517197 |
| 40 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000795.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 364.490 | 240.224 | 172.776 | 303.492 | 0.008404 | 0.013695 | 0.517533 |
| 41 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000815.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.035 | 240.610 | 170.423 | 303.288 | 0.008606 | 0.014145 | 0.521358 |
| 42 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000835.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.524 | 239.709 | 172.689 | 302.262 | 0.008741 | 0.014536 | 0.521851 |
| 43 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000856.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 359.507 | 239.062 | 174.753 | 298.169 | 0.008992 | 0.013530 | 0.518112 |
| 44 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000876.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 372.381 | 240.668 | 170.086 | 299.565 | 0.008013 | 0.014669 | 0.519884 |
| 45 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000897.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 363.815 | 236.587 | 169.972 | 297.844 | 0.008275 | 0.014335 | 0.521561 |
| 46 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000917.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 361.843 | 234.344 | 169.815 | 299.826 | 0.007720 | 0.013767 | 0.517622 |
| 47 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000937.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.621 | 236.358 | 169.870 | 300.205 | 0.009200 | 0.014796 | 0.519466 |
| 48 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000958.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 366.958 | 243.798 | 174.363 | 299.549 | 0.008581 | 0.013908 | 0.519274 |
| 49 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000978.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.515 | 238.905 | 170.281 | 300.841 | 0.008794 | 0.014147 | 0.518511 |
| 50 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000999.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 365.067 | 242.161 | 170.270 | 297.752 | 0.008474 | 0.013793 | 0.519899 |

## Raw Results
- `/root/autodl-tmp/wall_x/workspace/v4_video_pruning_compare/libero_50video_vispruner_on_v5_results.json`