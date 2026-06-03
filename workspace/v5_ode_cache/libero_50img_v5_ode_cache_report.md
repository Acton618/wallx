# Wall-X V5 ODE Cache Image Dataset Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `50`
- vispruner_enable: `True`
- keep_ratio: `0.5`
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

| case | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | cache_refreshes | cache_hits | cache_hit_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 345.610 | +0.00% | 284.858 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 220.102 | -36.31% | 159.320 | -44.07% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008379 | 0.010504 | 0.035543 |
| `cache_i3` | 157.267 | -54.50% | 96.312 | -66.19% | 10.00 | 3.00 | 6.00 | 66.67% | 0.014092 | 0.017676 | 0.056013 |
| `early_tradeoff` | 285.779 | -17.31% | 224.508 | -21.19% | 8.00 | 7.00 | 0.00 | 0.00% | 0.517705 | 0.662411 | 1.483980 |

## Interpretation

- `fixed_10` is the exact no-cache baseline with the original 10 Euler updates.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is reported as action difference against `fixed_10` under the same sample and seed.
- Fine-grained timings use `profile_timing=True`, so absolute numbers include CUDA event synchronization overhead; paired deltas are the useful signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 345.610 | 220.102 | 157.267 | 285.779 | -36.31% |
| `external_prepare_batch_ms` | 3.904 | 3.891 | 3.810 | 4.014 | -0.34% |
| `embed_processing` | 29.713 | 29.694 | 29.735 | 29.927 | -0.06% |
| `image_path_total` | 29.319 | 29.304 | 29.352 | 29.519 | -0.05% |
| `vision_image_forward` | 29.347 | 29.331 | 29.379 | 29.547 | -0.05% |
| `position_encoding` | 0.114 | 0.112 | 0.110 | 0.118 | -1.27% |
| `action_initialization` | 0.436 | 0.442 | 0.435 | 0.452 | +1.47% |
| `prefetch_forward` | 29.050 | 29.094 | 29.248 | 29.320 | +0.15% |
| `prefill_transformer` | 28.850 | 28.893 | 29.048 | 29.114 | +0.15% |
| `cache_preprocessing` | 1.305 | 1.305 | 1.295 | 1.318 | +0.00% |
| `ode_integration` | 284.858 | 159.320 | 96.312 | 224.508 | -44.07% |
| `ode_transformer_total` | 279.130 | 155.946 | 94.113 | 219.693 | -44.13% |
| `ode_action_embed_total` | 2.714 | 1.520 | 0.919 | 2.152 | -44.00% |
| `ode_prepare_inputs` | 0.652 | 0.367 | 0.219 | 0.519 | -43.79% |
| `ode_action_head_total` | 0.964 | 0.542 | 0.328 | 0.773 | -43.82% |
| `postprocessing` | 0.007 | 0.007 | 0.006 | 0.007 | -0.92% |
| `action_init_embed` | 0.257 | 0.263 | 0.259 | 0.266 | +1.98% |
| `action_init_noise` | 0.047 | 0.046 | 0.044 | 0.050 | -2.27% |
| `attention_mask_to_device` | 0.006 | 0.006 | 0.007 | 0.007 | +0.09% |
| `embed_tokens` | 0.042 | 0.040 | 0.038 | 0.046 | -4.53% |
| `image_cast` | 0.045 | 0.043 | 0.040 | 0.049 | -5.53% |
| `kv_cache_trim` | 0.852 | 0.856 | 0.855 | 0.854 | +0.53% |
| `moe_indices` | 0.092 | 0.091 | 0.089 | 0.096 | -1.71% |
| `ode_cache_hit` | 0.000 | 0.024 | 0.036 | 0.000 | +0.00% |
| `ode_cache_refresh` | 284.344 | 158.845 | 95.862 | 223.814 | -44.14% |
| `postfix_mask_build` | 0.134 | 0.131 | 0.126 | 0.139 | -2.66% |
| `postfix_moe_indices` | 0.102 | 0.103 | 0.102 | 0.105 | +0.54% |
| `postfix_slice` | 0.057 | 0.057 | 0.058 | 0.057 | +0.02% |
| `prefill_action_head` | 0.145 | 0.146 | 0.145 | 0.148 | +0.65% |
| `prefix_length_resolve` | 0.073 | 0.072 | 0.067 | 0.076 | -2.54% |
| `pruning_position_ids_prepare` | 0.614 | 0.604 | 0.593 | 0.643 | -1.55% |
| `scatter_action_init` | 0.058 | 0.058 | 0.058 | 0.059 | +1.12% |
| `scatter_image_embeds` | 0.105 | 0.103 | 0.100 | 0.109 | -1.76% |
| `scatter_proprioception` | 0.125 | 0.125 | 0.124 | 0.130 | +0.33% |
| `vision_image_encode_score` | 27.660 | 27.670 | 27.734 | 27.805 | +0.04% |
| `vispruner_apply_keep_to_sequences` | 0.199 | 0.199 | 0.199 | 0.202 | +0.29% |
| `vispruner_build_keep_mask` | 0.244 | 0.236 | 0.233 | 0.253 | -3.07% |
| `vispruner_gather_image_embeds` | 0.043 | 0.043 | 0.044 | 0.043 | +0.83% |
| `vispruner_image_lengths` | 0.079 | 0.077 | 0.075 | 0.080 | -3.14% |
| `vispruner_pad_pruned_batch` | 0.075 | 0.074 | 0.076 | 0.076 | -1.32% |
| `vispruner_rope_deltas` | 0.126 | 0.125 | 0.124 | 0.128 | -0.57% |
| `vispruner_score_prepare` | 0.044 | 0.041 | 0.039 | 0.047 | -5.97% |
| `vispruner_topk_select` | 0.130 | 0.126 | 0.124 | 0.137 | -3.04% |
| `vispruner_total` | 0.871 | 0.859 | 0.859 | 0.889 | -1.36% |

## Per-Sample Paired Results

| idx | source | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 354.222 | 217.447 | 157.106 | 283.103 | 0.008169 | 0.014392 | 0.521132 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 346.606 | 218.421 | 161.212 | 286.133 | 0.008490 | 0.013806 | 0.518725 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 343.072 | 216.787 | 161.720 | 283.305 | 0.008608 | 0.014326 | 0.518370 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 348.239 | 216.791 | 161.559 | 284.879 | 0.008208 | 0.013567 | 0.518436 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 347.963 | 217.573 | 160.397 | 283.881 | 0.008289 | 0.013701 | 0.518684 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 348.971 | 220.022 | 160.917 | 284.040 | 0.008327 | 0.014032 | 0.517734 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 351.842 | 218.511 | 161.678 | 282.233 | 0.008393 | 0.013895 | 0.518710 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 346.485 | 219.990 | 161.761 | 283.599 | 0.008221 | 0.014300 | 0.519002 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 343.964 | 222.980 | 158.947 | 283.786 | 0.008370 | 0.014031 | 0.523267 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 347.615 | 222.411 | 158.862 | 284.047 | 0.008561 | 0.013801 | 0.515129 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 346.801 | 223.123 | 156.357 | 282.851 | 0.008461 | 0.014455 | 0.517589 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 348.198 | 221.638 | 158.330 | 288.995 | 0.008489 | 0.013494 | 0.516511 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 345.450 | 223.018 | 156.737 | 283.918 | 0.008310 | 0.013560 | 0.517997 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 343.586 | 221.645 | 159.136 | 291.015 | 0.008634 | 0.014357 | 0.517867 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 342.836 | 221.913 | 156.462 | 290.303 | 0.008421 | 0.013836 | 0.517838 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 347.846 | 221.262 | 157.200 | 296.786 | 0.008119 | 0.014608 | 0.517360 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 348.542 | 219.935 | 156.687 | 285.443 | 0.008400 | 0.014550 | 0.517805 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 347.873 | 220.299 | 158.161 | 288.681 | 0.008235 | 0.013480 | 0.517606 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 345.134 | 220.533 | 157.634 | 289.857 | 0.008623 | 0.014255 | 0.516871 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 341.680 | 217.803 | 154.032 | 280.210 | 0.008307 | 0.013917 | 0.515767 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 349.866 | 222.642 | 163.953 | 287.551 | 0.008539 | 0.014811 | 0.516500 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 345.591 | 220.469 | 162.201 | 287.545 | 0.008699 | 0.013590 | 0.518103 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 350.083 | 221.714 | 159.819 | 287.246 | 0.008608 | 0.015260 | 0.518154 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 348.581 | 220.136 | 164.295 | 286.893 | 0.008429 | 0.014117 | 0.516216 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 349.509 | 219.984 | 156.423 | 287.184 | 0.008523 | 0.014200 | 0.516544 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 350.104 | 221.929 | 162.792 | 290.519 | 0.008781 | 0.014815 | 0.516617 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 349.102 | 219.959 | 160.232 | 286.173 | 0.008310 | 0.014061 | 0.516529 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 344.450 | 219.501 | 157.478 | 283.588 | 0.008557 | 0.014332 | 0.516530 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 337.687 | 215.346 | 157.350 | 277.801 | 0.008167 | 0.014487 | 0.514578 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 338.538 | 214.908 | 155.310 | 277.121 | 0.008498 | 0.013504 | 0.516567 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 337.449 | 214.850 | 156.881 | 277.155 | 0.008802 | 0.014108 | 0.515172 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 354.657 | 215.400 | 151.941 | 277.758 | 0.008488 | 0.014242 | 0.516339 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 342.203 | 221.706 | 153.440 | 282.346 | 0.007862 | 0.013557 | 0.517761 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 346.037 | 221.637 | 153.483 | 281.547 | 0.008244 | 0.013972 | 0.518302 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 344.731 | 219.413 | 154.047 | 285.635 | 0.008088 | 0.014091 | 0.515448 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 341.269 | 224.147 | 154.659 | 286.936 | 0.008813 | 0.014267 | 0.519871 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 343.070 | 221.619 | 154.167 | 288.939 | 0.008312 | 0.013607 | 0.517614 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 343.905 | 220.355 | 154.038 | 287.472 | 0.008517 | 0.014767 | 0.519426 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 345.621 | 220.975 | 153.728 | 289.526 | 0.008410 | 0.013813 | 0.520141 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 347.670 | 221.706 | 153.726 | 288.097 | 0.007928 | 0.014214 | 0.519612 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 346.967 | 217.416 | 155.056 | 287.698 | 0.008442 | 0.015176 | 0.520599 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 345.385 | 216.548 | 153.670 | 286.251 | 0.008607 | 0.013245 | 0.516469 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 344.913 | 216.930 | 153.657 | 288.324 | 0.008176 | 0.014224 | 0.517253 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 344.622 | 218.406 | 154.130 | 287.545 | 0.007849 | 0.013920 | 0.517564 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 344.113 | 218.628 | 154.037 | 289.057 | 0.008664 | 0.013846 | 0.520496 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 336.510 | 214.964 | 154.121 | 290.403 | 0.008371 | 0.014471 | 0.516368 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 343.430 | 225.793 | 155.320 | 291.124 | 0.008108 | 0.013755 | 0.518420 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 342.547 | 222.720 | 156.213 | 283.976 | 0.008504 | 0.014220 | 0.515893 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 341.690 | 226.322 | 157.463 | 286.578 | 0.008061 | 0.013762 | 0.516378 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 343.285 | 226.887 | 154.825 | 283.903 | 0.007952 | 0.013797 | 0.517380 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/v5_ode_cache/libero_50img_v5_ode_cache_results.json`