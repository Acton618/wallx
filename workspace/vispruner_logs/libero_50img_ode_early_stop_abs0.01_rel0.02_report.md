# Wall-X ODE Image Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `50`
- vispruner_enable: `True`
- keep_ratio: `0.5`
- num_inference_timesteps: `10`
- modified_mode: `early_stop`
- early_stop_patience: `1`
- early_stop_abs_threshold: `0.01`
- early_stop_rel_threshold: `0.02`
- warmup: `1`
- iters: `3`
- device: `cuda`

## Summary

- samples: `50`
- tokens: `81.00` -> `41.00`
- original total_time: `339.932 ms`
- modified total_time: `339.417 ms`
- total_time delta: `-0.516 ms` (`-0.15%`)
- original ode_integration: `280.949 ms`
- modified ode_integration: `280.669 ms`
- ode_integration delta: `-0.280 ms` (`-0.10%`)
- original ode_steps_used: `9.00`
- modified ode_steps_used: `9.00`
- action MAE vs original: `0.004703`
- action RMSE vs original: `0.005908`
- action max_abs vs original: `0.019801`

## Stage Timing

| stage | original_ms | modified_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 339.932 | 339.417 | -0.516 | -0.15% |
| `external_prepare_batch_ms` | 3.475 | 3.420 | -0.055 | -1.59% |
| `embed_processing` | 27.151 | 27.101 | -0.051 | -0.19% |
| `image_path_total` | 26.752 | 26.703 | -0.049 | -0.18% |
| `vision_image_forward` | 26.780 | 26.731 | -0.048 | -0.18% |
| `vision_image_encode_score` | 25.162 | 25.106 | -0.055 | -0.22% |
| `scatter_image_embeds` | 0.107 | 0.106 | -0.000 | -0.30% |
| `position_encoding` | 0.113 | 0.113 | -0.001 | -0.64% |
| `action_initialization` | 0.442 | 0.439 | -0.003 | -0.77% |
| `prefetch_forward` | 29.872 | 29.678 | -0.195 | -0.65% |
| `prefill_transformer` | 29.667 | 29.476 | -0.191 | -0.64% |
| `cache_preprocessing` | 1.267 | 1.279 | +0.012 | +0.95% |
| `ode_integration` | 280.949 | 280.669 | -0.280 | -0.10% |
| `ode_action_embed_total` | 2.689 | 2.714 | +0.025 | +0.91% |
| `ode_prepare_inputs` | 0.644 | 0.651 | +0.007 | +1.09% |
| `ode_transformer_total` | 274.470 | 274.445 | -0.025 | -0.01% |
| `ode_action_head_total` | 0.976 | 0.980 | +0.004 | +0.36% |
| `postprocessing` | 0.006 | 0.007 | +0.000 | +2.24% |
| `action_init_embed` | 0.262 | 0.260 | -0.002 | -0.72% |
| `action_init_noise` | 0.049 | 0.049 | -0.000 | -0.12% |
| `attention_mask_to_device` | 0.006 | 0.006 | -0.000 | -0.88% |
| `embed_tokens` | 0.044 | 0.044 | -0.000 | -0.15% |
| `image_cast` | 0.049 | 0.049 | +0.000 | +0.01% |
| `kv_cache_trim` | 0.811 | 0.817 | +0.006 | +0.72% |
| `moe_indices` | 0.092 | 0.092 | -0.001 | -0.61% |
| `postfix_mask_build` | 0.132 | 0.135 | +0.002 | +1.86% |
| `postfix_moe_indices` | 0.105 | 0.107 | +0.002 | +1.83% |
| `postfix_slice` | 0.054 | 0.056 | +0.002 | +3.03% |
| `prefill_action_head` | 0.149 | 0.146 | -0.002 | -1.55% |
| `prefix_length_resolve` | 0.077 | 0.076 | -0.000 | -0.48% |
| `pruning_position_ids_prepare` | 0.556 | 0.560 | +0.004 | +0.69% |
| `scatter_action_init` | 0.057 | 0.056 | -0.001 | -1.35% |
| `scatter_proprioception` | 0.128 | 0.126 | -0.001 | -0.98% |
| `vispruner_apply_keep_to_sequences` | 0.195 | 0.196 | +0.001 | +0.43% |
| `vispruner_build_keep_mask` | 0.240 | 0.242 | +0.002 | +1.00% |
| `vispruner_gather_image_embeds` | 0.040 | 0.041 | +0.001 | +1.78% |
| `vispruner_image_lengths` | 0.065 | 0.064 | -0.000 | -0.74% |
| `vispruner_pad_pruned_batch` | 0.076 | 0.075 | -0.001 | -1.03% |
| `vispruner_rope_deltas` | 0.129 | 0.128 | -0.001 | -0.78% |
| `vispruner_score_prepare` | 0.031 | 0.031 | +0.000 | +0.06% |
| `vispruner_topk_select` | 0.133 | 0.135 | +0.002 | +1.33% |
| `vispruner_total` | 0.850 | 0.853 | +0.003 | +0.31% |

## Paired Samples

| idx | source | tokens | original_steps | modified_steps | original_total_ms | modified_total_ms | total_delta_pct | original_ode_ms | modified_ode_ms | ode_delta_pct | action_mae | action_rmse | action_max_abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 343.411 | 335.531 | -2.29% | 284.268 | 277.612 | -2.34% | 0.004736 | 0.005900 | 0.017990 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 342.134 | 334.932 | -2.11% | 282.993 | 277.027 | -2.11% | 0.004836 | 0.006063 | 0.020499 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.045 | 337.147 | -2.00% | 285.405 | 278.970 | -2.25% | 0.004751 | 0.005911 | 0.020555 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.088 | 335.616 | -1.31% | 281.479 | 277.836 | -1.29% | 0.004454 | 0.005615 | 0.018546 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 348.105 | 337.490 | -3.05% | 284.885 | 279.088 | -2.04% | 0.004773 | 0.006064 | 0.021203 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.415 | 336.946 | -2.17% | 285.446 | 277.847 | -2.66% | 0.004853 | 0.006067 | 0.020969 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.072 | 339.705 | -1.27% | 285.275 | 279.887 | -1.89% | 0.004924 | 0.006155 | 0.022814 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.863 | 340.734 | -1.20% | 285.585 | 282.664 | -1.02% | 0.004770 | 0.005943 | 0.018163 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.644 | 340.091 | -0.16% | 281.478 | 281.480 | +0.00% | 0.004551 | 0.005731 | 0.017377 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.177 | 342.752 | +0.46% | 282.313 | 284.323 | +0.71% | 0.004417 | 0.005574 | 0.017667 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 342.923 | 341.697 | -0.36% | 283.912 | 282.796 | -0.39% | 0.004712 | 0.005942 | 0.018536 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.487 | 339.403 | +0.27% | 280.204 | 281.416 | +0.43% | 0.004531 | 0.005724 | 0.020483 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.261 | 340.834 | +0.17% | 280.537 | 281.851 | +0.47% | 0.004914 | 0.006110 | 0.019316 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.286 | 338.412 | +0.04% | 279.913 | 280.257 | +0.12% | 0.004588 | 0.005766 | 0.018539 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.018 | 341.539 | +0.74% | 280.473 | 283.529 | +1.09% | 0.004709 | 0.005959 | 0.020882 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.175 | 340.146 | -0.01% | 280.924 | 280.701 | -0.08% | 0.004838 | 0.006067 | 0.018862 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.985 | 337.977 | -0.30% | 280.429 | 279.192 | -0.44% | 0.004466 | 0.005537 | 0.016715 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.414 | 339.702 | -1.65% | 285.015 | 280.547 | -1.57% | 0.004246 | 0.005347 | 0.019282 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.289 | 340.418 | -0.26% | 282.223 | 280.810 | -0.50% | 0.004781 | 0.006084 | 0.019559 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 336.001 | 337.259 | +0.37% | 277.291 | 278.812 | +0.55% | 0.004732 | 0.005952 | 0.018605 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.907 | 340.957 | -0.28% | 282.235 | 281.981 | -0.09% | 0.004518 | 0.005655 | 0.017456 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.434 | 343.700 | +1.26% | 281.036 | 284.969 | +1.40% | 0.004568 | 0.005697 | 0.017000 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.244 | 343.238 | +0.88% | 281.355 | 283.979 | +0.93% | 0.004711 | 0.005842 | 0.016713 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.214 | 340.381 | +0.34% | 280.766 | 282.181 | +0.50% | 0.004733 | 0.006020 | 0.020338 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.231 | 344.823 | +1.05% | 282.075 | 285.395 | +1.18% | 0.004527 | 0.005823 | 0.018737 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.597 | 339.662 | -0.57% | 282.141 | 281.527 | -0.22% | 0.004606 | 0.005740 | 0.020181 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.360 | 338.396 | -0.87% | 282.631 | 280.324 | -0.82% | 0.004507 | 0.005736 | 0.019446 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 343.817 | 341.623 | -0.64% | 283.589 | 283.627 | +0.01% | 0.004804 | 0.005999 | 0.022059 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.732 | 336.076 | +0.40% | 275.904 | 277.291 | +0.50% | 0.004759 | 0.006003 | 0.021463 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.607 | 340.961 | +0.70% | 277.718 | 278.411 | +0.25% | 0.004910 | 0.006163 | 0.018435 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.307 | 335.341 | -1.17% | 278.827 | 276.816 | -0.72% | 0.004966 | 0.006161 | 0.022416 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.819 | 335.802 | -0.89% | 278.991 | 276.968 | -0.72% | 0.004771 | 0.005914 | 0.021114 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 342.699 | 352.816 | +2.95% | 284.225 | 294.510 | +3.62% | 0.004490 | 0.005582 | 0.019478 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.216 | 340.750 | +0.75% | 279.623 | 281.428 | +0.65% | 0.004748 | 0.006062 | 0.026895 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.631 | 340.389 | +0.52% | 280.317 | 282.158 | +0.66% | 0.004881 | 0.006072 | 0.018281 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 337.248 | 339.232 | +0.59% | 279.529 | 280.171 | +0.23% | 0.004748 | 0.005990 | 0.021463 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 336.187 | 341.049 | +1.45% | 278.047 | 282.273 | +1.52% | 0.004495 | 0.005622 | 0.020296 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 335.957 | 341.355 | +1.61% | 277.701 | 281.965 | +1.54% | 0.004851 | 0.006023 | 0.019043 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 335.804 | 341.277 | +1.63% | 277.702 | 282.266 | +1.64% | 0.004783 | 0.005956 | 0.016944 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.117 | 341.626 | +1.04% | 279.000 | 281.519 | +0.90% | 0.004659 | 0.005802 | 0.018767 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 338.571 | 338.988 | +0.12% | 280.108 | 279.350 | -0.27% | 0.004749 | 0.005922 | 0.020370 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.126 | 335.052 | -1.49% | 281.987 | 277.245 | -1.68% | 0.004489 | 0.005717 | 0.016250 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 342.198 | 338.663 | -1.03% | 283.818 | 279.881 | -1.39% | 0.005108 | 0.006342 | 0.022980 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.114 | 336.097 | -1.18% | 281.436 | 278.098 | -1.19% | 0.004748 | 0.006081 | 0.025501 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 342.844 | 338.233 | -1.35% | 281.413 | 279.233 | -0.77% | 0.004818 | 0.006000 | 0.018897 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 330.962 | 330.521 | -0.13% | 272.398 | 272.085 | -0.11% | 0.004672 | 0.005817 | 0.018787 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 335.836 | 339.015 | +0.95% | 277.718 | 280.732 | +1.09% | 0.004911 | 0.006270 | 0.024894 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 336.249 | 339.326 | +0.92% | 278.364 | 280.955 | +0.93% | 0.004585 | 0.005879 | 0.018686 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 335.411 | 337.419 | +0.60% | 277.590 | 279.164 | +0.57% | 0.004585 | 0.005877 | 0.021232 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 337.390 | 339.740 | +0.70% | 279.151 | 280.309 | +0.41% | 0.004862 | 0.006121 | 0.019379 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_early_stop_abs0.01_rel0.02_results.json`