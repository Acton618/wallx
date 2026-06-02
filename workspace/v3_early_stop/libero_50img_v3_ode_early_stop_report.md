# Wall-X V3 ODE Early Stop Dataset Report

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

## V3 Cases

| case | enable | threshold | min_steps | patience | metric |
|---|---:|---:|---:|---:|---|
| `fixed_10` | `False` | `-` | `-` | `-` | `mean_abs` |
| `early_safe` | `True` | `0.2` | `2` | `1` | `mean_abs` |
| `early_tradeoff` | `True` | `0.3` | `8` | `1` | `mean_abs` |

## Summary

| case | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | postfix_steps | stopped_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 342.222 | +0.00% | 282.213 | +0.00% | 10.00 | 9.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `early_safe` | 332.265 | -2.91% | 274.404 | -2.77% | 10.00 | 9.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `early_tradeoff` | 271.367 | -20.70% | 213.471 | -24.36% | 8.00 | 7.00 | 100.00% | 0.517705 | 0.662411 | 1.483980 |

## Interpretation

- `fixed_10` is the V3-compatible baseline with early stop disabled; it preserves original fixed-step behavior.
- `actual_updates` counts the existing prefetch update plus later postfix ODE updates. `postfix_steps` is `actual_updates - 1`.
- Accuracy is reported as action difference against `fixed_10` under the same sample and seed, because this benchmark measures whether V3 early stop changes the original model output.
- Fine-grained timings use `profile_timing=True`, so absolute numbers include CUDA event synchronization overhead; paired deltas are the useful signal.

## Stage Timing

| stage | fixed_10_ms | early_safe_ms | early_tradeoff_ms | tradeoff_delta_vs_fixed |
|---|---:|---:|---:|---:|
| `total_time` | 342.222 | 332.265 | 271.367 | -20.70% |
| `external_prepare_batch_ms` | 4.431 | 4.206 | 4.227 | -4.59% |
| `embed_processing` | 28.987 | 28.068 | 28.065 | -3.18% |
| `image_path_total` | 28.586 | 27.687 | 27.676 | -3.18% |
| `vision_image_forward` | 28.614 | 27.713 | 27.703 | -3.18% |
| `position_encoding` | 0.114 | 0.112 | 0.115 | +0.94% |
| `action_initialization` | 0.437 | 0.430 | 0.431 | -1.51% |
| `prefetch_forward` | 29.048 | 27.867 | 27.900 | -3.95% |
| `prefill_transformer` | 28.839 | 27.669 | 27.704 | -3.93% |
| `cache_preprocessing` | 1.287 | 1.254 | 1.253 | -2.69% |
| `ode_integration` | 282.213 | 274.404 | 213.471 | -24.36% |
| `ode_transformer_total` | 275.710 | 268.638 | 209.011 | -24.19% |
| `ode_action_embed_total` | 2.686 | 2.658 | 2.073 | -22.83% |
| `ode_prepare_inputs` | 0.647 | 0.632 | 0.492 | -24.00% |
| `ode_action_head_total` | 0.986 | 0.957 | 0.746 | -24.33% |
| `postprocessing` | 0.006 | 0.006 | 0.006 | -5.35% |
| `action_init_embed` | 0.258 | 0.256 | 0.256 | -0.77% |
| `action_init_noise` | 0.049 | 0.046 | 0.047 | -5.03% |
| `attention_mask_to_device` | 0.006 | 0.006 | 0.006 | +2.48% |
| `embed_tokens` | 0.046 | 0.040 | 0.040 | -12.41% |
| `image_cast` | 0.049 | 0.045 | 0.043 | -11.28% |
| `kv_cache_trim` | 0.829 | 0.814 | 0.815 | -1.64% |
| `moe_indices` | 0.093 | 0.091 | 0.092 | -0.07% |
| `postfix_mask_build` | 0.137 | 0.129 | 0.128 | -7.18% |
| `postfix_moe_indices` | 0.103 | 0.101 | 0.101 | -1.86% |
| `postfix_slice` | 0.055 | 0.054 | 0.054 | -1.42% |
| `prefill_action_head` | 0.152 | 0.143 | 0.143 | -6.24% |
| `prefix_length_resolve` | 0.076 | 0.070 | 0.069 | -9.04% |
| `pruning_position_ids_prepare` | 0.616 | 0.597 | 0.589 | -4.51% |
| `scatter_action_init` | 0.057 | 0.056 | 0.056 | -2.74% |
| `scatter_image_embeds` | 0.108 | 0.102 | 0.103 | -4.18% |
| `scatter_proprioception` | 0.127 | 0.122 | 0.125 | -1.33% |
| `vision_image_encode_score` | 26.923 | 26.075 | 26.082 | -3.12% |
| `vispruner_apply_keep_to_sequences` | 0.194 | 0.189 | 0.189 | -2.34% |
| `vispruner_build_keep_mask` | 0.252 | 0.239 | 0.235 | -6.54% |
| `vispruner_gather_image_embeds` | 0.042 | 0.041 | 0.041 | -1.62% |
| `vispruner_image_lengths` | 0.078 | 0.075 | 0.075 | -4.01% |
| `vispruner_pad_pruned_batch` | 0.075 | 0.073 | 0.072 | -4.16% |
| `vispruner_rope_deltas` | 0.124 | 0.121 | 0.121 | -2.27% |
| `vispruner_score_prepare` | 0.048 | 0.044 | 0.042 | -12.10% |
| `vispruner_topk_select` | 0.135 | 0.128 | 0.126 | -7.11% |
| `vispruner_total` | 0.868 | 0.842 | 0.836 | -3.61% |

## Per-Sample Paired Results

| idx | source | safe_updates | tradeoff_updates | fixed_total_ms | safe_total_ms | tradeoff_total_ms | safe_mae | tradeoff_mae |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 10.00 | 8.00 | 343.197 | 333.398 | 276.337 | 0.000000 | 0.521132 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 10.00 | 8.00 | 344.042 | 328.737 | 276.919 | 0.000000 | 0.518725 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 10.00 | 8.00 | 343.068 | 329.320 | 274.386 | 0.000000 | 0.518370 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.161 | 330.263 | 269.758 | 0.000000 | 0.518436 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 10.00 | 8.00 | 342.511 | 330.477 | 270.121 | 0.000000 | 0.518684 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 10.00 | 8.00 | 342.666 | 331.150 | 271.906 | 0.000000 | 0.517734 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 10.00 | 8.00 | 341.740 | 332.608 | 269.959 | 0.000000 | 0.518710 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 10.00 | 8.00 | 345.863 | 332.509 | 272.301 | 0.000000 | 0.519002 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.504 | 333.181 | 272.440 | 0.000000 | 0.523267 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.309 | 331.656 | 274.125 | 0.000000 | 0.515129 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.883 | 334.083 | 271.770 | 0.000000 | 0.517589 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.063 | 333.086 | 272.041 | 0.000000 | 0.516511 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.349 | 332.896 | 269.507 | 0.000000 | 0.517997 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 10.00 | 8.00 | 343.215 | 337.041 | 270.614 | 0.000000 | 0.517867 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 10.00 | 8.00 | 340.029 | 334.310 | 270.669 | 0.000000 | 0.517838 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 10.00 | 8.00 | 340.564 | 334.935 | 271.140 | 0.000000 | 0.517360 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.408 | 333.486 | 271.042 | 0.000000 | 0.517805 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.010 | 334.336 | 271.688 | 0.000000 | 0.517606 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 10.00 | 8.00 | 340.208 | 332.013 | 272.557 | 0.000000 | 0.516871 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 10.00 | 8.00 | 332.882 | 328.139 | 267.105 | 0.000000 | 0.515767 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 10.00 | 8.00 | 343.027 | 333.807 | 276.383 | 0.000000 | 0.516500 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 10.00 | 8.00 | 365.919 | 330.030 | 274.735 | 0.000000 | 0.518103 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 10.00 | 8.00 | 365.022 | 335.734 | 270.429 | 0.000000 | 0.518154 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 10.00 | 8.00 | 362.137 | 333.926 | 271.123 | 0.000000 | 0.516216 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 10.00 | 8.00 | 365.812 | 333.127 | 270.750 | 0.000000 | 0.516544 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 10.00 | 8.00 | 344.866 | 330.605 | 270.520 | 0.000000 | 0.516617 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.504 | 332.129 | 271.143 | 0.000000 | 0.516529 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 10.00 | 8.00 | 344.439 | 332.015 | 272.653 | 0.000000 | 0.516530 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 10.00 | 8.00 | 335.747 | 327.599 | 268.047 | 0.000000 | 0.514578 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 10.00 | 8.00 | 338.134 | 326.516 | 267.877 | 0.000000 | 0.516567 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 10.00 | 8.00 | 338.892 | 330.512 | 267.553 | 0.000000 | 0.515172 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 10.00 | 8.00 | 335.713 | 325.617 | 268.207 | 0.000000 | 0.516339 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 10.00 | 8.00 | 341.256 | 329.972 | 270.691 | 0.000000 | 0.517761 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 10.00 | 8.00 | 341.746 | 332.657 | 270.363 | 0.000000 | 0.518302 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 10.00 | 8.00 | 345.053 | 334.900 | 270.328 | 0.000000 | 0.515448 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 10.00 | 8.00 | 344.839 | 330.894 | 270.343 | 0.000000 | 0.519871 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 10.00 | 8.00 | 339.689 | 332.662 | 271.273 | 0.000000 | 0.517614 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 10.00 | 8.00 | 342.161 | 334.515 | 271.861 | 0.000000 | 0.519426 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 10.00 | 8.00 | 343.650 | 332.558 | 271.178 | 0.000000 | 0.520141 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 10.00 | 8.00 | 341.257 | 332.166 | 271.548 | 0.000000 | 0.519612 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 10.00 | 8.00 | 344.125 | 333.563 | 271.359 | 0.000000 | 0.520599 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 10.00 | 8.00 | 344.048 | 331.149 | 271.921 | 0.000000 | 0.516469 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 10.00 | 8.00 | 351.802 | 334.039 | 270.513 | 0.000000 | 0.517253 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 10.00 | 8.00 | 335.331 | 331.965 | 271.807 | 0.000000 | 0.517564 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 10.00 | 8.00 | 331.701 | 332.369 | 271.243 | 0.000000 | 0.520496 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 10.00 | 8.00 | 327.772 | 326.905 | 266.339 | 0.000000 | 0.516368 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 10.00 | 8.00 | 333.912 | 334.312 | 272.898 | 0.000000 | 0.518420 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 10.00 | 8.00 | 337.510 | 336.141 | 271.845 | 0.000000 | 0.515893 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 10.00 | 8.00 | 336.773 | 336.283 | 274.460 | 0.000000 | 0.516378 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 10.00 | 8.00 | 334.595 | 336.959 | 272.583 | 0.000000 | 0.517380 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_v3_ode_early_stop_results.json`