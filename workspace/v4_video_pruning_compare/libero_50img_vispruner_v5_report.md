# Wall-X V5 ODE Cache Image Dataset Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `50`
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
| `fixed_10` | 341.600 | +0.00% | 281.455 | +0.00% | 10.00 | 9.00 | 0.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `cache_i2` | 217.309 | -36.38% | 157.472 | -44.05% | 10.00 | 5.00 | 4.00 | 44.44% | 0.008434 | 0.010595 | 0.034757 |
| `cache_i3` | 155.518 | -54.47% | 95.497 | -66.07% | 10.00 | 3.00 | 6.00 | 66.67% | 0.014082 | 0.017653 | 0.054655 |
| `early_tradeoff` | 281.224 | -17.67% | 221.347 | -21.36% | 8.00 | 7.00 | 0.00 | 0.00% | 0.517708 | 0.662466 | 1.485309 |

## Interpretation

- `fixed_10` is the exact no-cache baseline with the original 10 Euler updates.
- V5 ODE cache reuses the previous velocity on cache-hit steps. It is disabled unless a case passes `ode_cache_enable=True`.
- Accuracy is reported as action difference against `fixed_10` under the same sample and seed.
- Fine-grained timings use `profile_timing=True`, so absolute numbers include CUDA event synchronization overhead; paired deltas are the useful signal.

## Stage Timing

| stage | fixed_10_ms | cache_i2_ms | cache_i3_ms | early_tradeoff_ms | cache_i2_delta_vs_fixed |
|---|---:|---:|---:|---:|---:|
| `total_time` | 341.600 | 217.309 | 155.518 | 281.224 | -36.38% |
| `external_prepare_batch_ms` | 4.039 | 3.689 | 3.676 | 3.644 | -8.67% |
| `embed_processing` | 29.009 | 29.053 | 29.106 | 29.067 | +0.15% |
| `image_path_total` | 28.532 | 28.658 | 28.715 | 28.678 | +0.44% |
| `vision_image_forward` | 28.559 | 28.686 | 28.742 | 28.704 | +0.44% |
| `position_encoding` | 0.115 | 0.115 | 0.111 | 0.112 | +0.20% |
| `action_initialization` | 0.527 | 0.443 | 0.437 | 0.438 | -15.90% |
| `prefetch_forward` | 28.918 | 28.795 | 28.900 | 28.816 | -0.43% |
| `prefill_transformer` | 28.718 | 28.597 | 28.694 | 28.612 | -0.42% |
| `cache_preprocessing` | 1.442 | 1.295 | 1.336 | 1.311 | -10.16% |
| `ode_integration` | 281.455 | 157.472 | 95.497 | 221.347 | -44.05% |
| `ode_transformer_total` | 275.792 | 154.118 | 93.296 | 216.561 | -44.12% |
| `ode_action_embed_total` | 2.702 | 1.511 | 0.921 | 2.118 | -44.08% |
| `ode_prepare_inputs` | 0.647 | 0.365 | 0.221 | 0.515 | -43.56% |
| `ode_action_head_total` | 0.956 | 0.535 | 0.326 | 0.760 | -44.07% |
| `postprocessing` | 0.007 | 0.007 | 0.006 | 0.007 | -6.65% |
| `action_init_embed` | 0.334 | 0.261 | 0.259 | 0.259 | -21.81% |
| `action_init_noise` | 0.061 | 0.048 | 0.047 | 0.047 | -20.09% |
| `attention_mask_to_device` | 0.006 | 0.006 | 0.006 | 0.006 | -0.63% |
| `embed_tokens` | 0.050 | 0.042 | 0.040 | 0.041 | -15.55% |
| `image_cast` | 0.040 | 0.038 | 0.037 | 0.037 | -4.45% |
| `kv_cache_trim` | 0.851 | 0.853 | 0.889 | 0.861 | +0.24% |
| `moe_indices` | 0.093 | 0.093 | 0.090 | 0.091 | -0.32% |
| `ode_cache_hit` | 0.000 | 0.025 | 0.036 | 0.000 | +0.00% |
| `ode_cache_refresh` | 280.947 | 157.001 | 95.048 | 220.624 | -44.12% |
| `postfix_mask_build` | 0.134 | 0.126 | 0.126 | 0.126 | -6.33% |
| `postfix_moe_indices` | 0.147 | 0.101 | 0.102 | 0.103 | -31.28% |
| `postfix_slice` | 0.057 | 0.057 | 0.057 | 0.058 | +0.49% |
| `prefill_action_head` | 0.146 | 0.144 | 0.149 | 0.148 | -1.44% |
| `prefix_length_resolve` | 0.167 | 0.072 | 0.073 | 0.071 | -56.69% |
| `pruning_position_ids_prepare` | 0.575 | 0.570 | 0.578 | 0.579 | -0.89% |
| `scatter_action_init` | 0.058 | 0.057 | 0.057 | 0.057 | -0.20% |
| `scatter_image_embeds` | 0.116 | 0.106 | 0.104 | 0.104 | -8.72% |
| `scatter_proprioception` | 0.190 | 0.125 | 0.125 | 0.125 | -34.26% |
| `vision_image_encode_score` | 26.952 | 27.095 | 27.141 | 27.105 | +0.53% |
| `vispruner_apply_keep_to_sequences` | 0.201 | 0.196 | 0.197 | 0.197 | -2.23% |
| `vispruner_build_keep_mask` | 0.226 | 0.222 | 0.225 | 0.224 | -1.85% |
| `vispruner_gather_image_embeds` | 0.042 | 0.043 | 0.043 | 0.043 | +1.05% |
| `vispruner_image_lengths` | 0.074 | 0.073 | 0.073 | 0.073 | -1.59% |
| `vispruner_pad_pruned_batch` | 0.073 | 0.072 | 0.072 | 0.073 | -1.16% |
| `vispruner_rope_deltas` | 0.122 | 0.121 | 0.123 | 0.123 | -0.80% |
| `vispruner_score_prepare` | 0.039 | 0.037 | 0.038 | 0.037 | -5.10% |
| `vispruner_topk_select` | 0.119 | 0.117 | 0.119 | 0.118 | -1.92% |
| `vispruner_total` | 0.842 | 0.830 | 0.836 | 0.836 | -1.36% |

## Per-Sample Paired Results

| idx | source | fixed_total_ms | cache_i2_total_ms | cache_i3_total_ms | early_tradeoff_total_ms | cache_i2_mae | cache_i3_mae | early_tradeoff_mae |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 370.366 | 216.472 | 156.560 | 283.640 | 0.008979 | 0.014979 | 0.520605 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 344.495 | 219.910 | 155.213 | 280.903 | 0.008394 | 0.014622 | 0.518650 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 343.504 | 218.181 | 153.213 | 282.833 | 0.008599 | 0.014600 | 0.518229 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 341.328 | 218.193 | 152.933 | 281.096 | 0.008403 | 0.013611 | 0.518875 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 341.164 | 217.838 | 153.970 | 283.712 | 0.008438 | 0.013929 | 0.517600 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 347.681 | 218.680 | 155.457 | 284.744 | 0.008797 | 0.013621 | 0.517871 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 341.064 | 220.153 | 157.436 | 282.821 | 0.008408 | 0.014140 | 0.518738 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 340.730 | 217.407 | 153.512 | 282.252 | 0.008222 | 0.014178 | 0.519176 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 342.937 | 216.974 | 158.543 | 283.091 | 0.009075 | 0.015069 | 0.522710 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 343.546 | 220.421 | 199.869 | 282.837 | 0.008993 | 0.013739 | 0.515543 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 348.254 | 217.076 | 154.500 | 283.413 | 0.008637 | 0.014223 | 0.518135 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 346.636 | 216.698 | 154.331 | 284.053 | 0.008370 | 0.013865 | 0.516099 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 347.330 | 217.051 | 153.853 | 283.709 | 0.008113 | 0.013904 | 0.517932 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 342.232 | 217.541 | 154.415 | 283.119 | 0.008191 | 0.014634 | 0.516648 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 339.676 | 221.196 | 154.755 | 281.883 | 0.008379 | 0.014648 | 0.517764 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 341.299 | 218.159 | 153.458 | 281.238 | 0.008113 | 0.014017 | 0.517187 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 340.843 | 219.361 | 154.913 | 280.445 | 0.008336 | 0.013375 | 0.518215 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 340.758 | 216.058 | 156.595 | 281.805 | 0.008651 | 0.013930 | 0.516934 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 340.073 | 217.583 | 153.774 | 283.567 | 0.008865 | 0.014517 | 0.516954 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 339.863 | 214.083 | 152.346 | 278.625 | 0.008607 | 0.013963 | 0.516614 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 342.659 | 216.344 | 153.471 | 283.121 | 0.008204 | 0.013917 | 0.516697 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 340.500 | 216.812 | 155.029 | 282.213 | 0.008022 | 0.014255 | 0.518535 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 341.967 | 216.514 | 154.022 | 280.163 | 0.008381 | 0.014497 | 0.517960 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 342.122 | 217.601 | 154.015 | 281.406 | 0.008530 | 0.013775 | 0.515451 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 339.038 | 219.013 | 153.732 | 279.570 | 0.008534 | 0.014290 | 0.516527 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 336.659 | 216.019 | 153.840 | 280.330 | 0.008487 | 0.014460 | 0.517480 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 339.821 | 215.746 | 154.014 | 281.257 | 0.008356 | 0.013915 | 0.516441 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 337.268 | 216.090 | 154.502 | 281.444 | 0.008321 | 0.014128 | 0.517185 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 330.998 | 213.344 | 151.510 | 277.866 | 0.008439 | 0.013575 | 0.515006 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 334.702 | 212.443 | 152.146 | 277.289 | 0.008092 | 0.014037 | 0.516271 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 332.146 | 213.769 | 151.723 | 275.457 | 0.008701 | 0.014523 | 0.515740 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 332.778 | 213.352 | 151.596 | 275.510 | 0.008127 | 0.013464 | 0.516934 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 337.867 | 218.679 | 155.530 | 281.388 | 0.008359 | 0.013779 | 0.517583 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 339.288 | 217.217 | 154.276 | 280.942 | 0.008375 | 0.014116 | 0.517929 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 338.141 | 219.132 | 154.914 | 281.826 | 0.008217 | 0.013226 | 0.515970 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 338.492 | 219.919 | 154.409 | 282.003 | 0.008602 | 0.013970 | 0.520476 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 341.615 | 217.504 | 154.381 | 281.350 | 0.008505 | 0.013797 | 0.517241 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 342.024 | 219.214 | 156.694 | 281.919 | 0.008144 | 0.014036 | 0.520004 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 341.534 | 222.214 | 155.532 | 280.823 | 0.008183 | 0.013968 | 0.519504 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 339.375 | 217.583 | 155.004 | 280.925 | 0.008484 | 0.014274 | 0.519501 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 346.382 | 216.185 | 154.928 | 280.290 | 0.008859 | 0.014257 | 0.520556 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 339.187 | 216.847 | 154.824 | 279.700 | 0.008485 | 0.013783 | 0.516672 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 342.077 | 218.014 | 155.495 | 281.443 | 0.008467 | 0.013863 | 0.517786 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 347.431 | 216.169 | 156.057 | 281.208 | 0.008457 | 0.013979 | 0.517184 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 341.131 | 218.468 | 155.929 | 281.507 | 0.008566 | 0.014010 | 0.520461 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 335.295 | 214.450 | 154.015 | 276.574 | 0.008099 | 0.013855 | 0.516451 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 340.079 | 216.175 | 157.290 | 281.105 | 0.007954 | 0.014220 | 0.518131 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 345.394 | 216.429 | 155.845 | 279.898 | 0.008230 | 0.013931 | 0.515225 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 344.226 | 217.553 | 156.188 | 280.313 | 0.008173 | 0.014408 | 0.517079 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 346.007 | 217.626 | 155.342 | 282.574 | 0.008761 | 0.014249 | 0.516937 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/v4_video_pruning_compare/libero_50img_vispruner_v5_results.json`