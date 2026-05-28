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
- early_stop_patience: `2`
- early_stop_abs_threshold: `0.002`
- early_stop_rel_threshold: `0.005`
- warmup: `1`
- iters: `3`
- device: `cuda`

## Summary

- samples: `50`
- tokens: `81.00` -> `41.00`
- original total_time: `332.851 ms`
- modified total_time: `334.862 ms`
- total_time delta: `+2.011 ms` (`+0.60%`)
- original ode_integration: `274.942 ms`
- modified ode_integration: `276.469 ms`
- ode_integration delta: `+1.527 ms` (`+0.56%`)
- original ode_steps_used: `9.00`
- modified ode_steps_used: `9.00`

## Stage Timing

| stage | original_ms | modified_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 332.851 | 334.862 | +2.011 | +0.60% |
| `external_prepare_batch_ms` | 3.557 | 3.553 | -0.003 | -0.09% |
| `embed_processing` | 26.762 | 26.963 | +0.201 | +0.75% |
| `image_path_total` | 26.372 | 26.559 | +0.187 | +0.71% |
| `vision_image_forward` | 26.399 | 26.586 | +0.187 | +0.71% |
| `vision_image_encode_score` | 24.801 | 24.955 | +0.154 | +0.62% |
| `scatter_image_embeds` | 0.104 | 0.108 | +0.003 | +3.12% |
| `position_encoding` | 0.112 | 0.116 | +0.005 | +4.18% |
| `action_initialization` | 0.434 | 0.456 | +0.023 | +5.21% |
| `prefetch_forward` | 29.206 | 29.449 | +0.243 | +0.83% |
| `prefill_transformer` | 29.003 | 29.240 | +0.237 | +0.82% |
| `cache_preprocessing` | 1.261 | 1.272 | +0.011 | +0.88% |
| `ode_integration` | 274.942 | 276.469 | +1.527 | +0.56% |
| `ode_action_embed_total` | 2.673 | 2.767 | +0.095 | +3.54% |
| `ode_prepare_inputs` | 0.632 | 0.647 | +0.015 | +2.35% |
| `ode_transformer_total` | 268.499 | 270.116 | +1.616 | +0.60% |
| `ode_action_head_total` | 0.971 | 1.020 | +0.049 | +5.06% |
| `postprocessing` | 0.006 | 0.006 | -0.000 | -1.03% |
| `action_init_embed` | 0.259 | 0.274 | +0.015 | +5.90% |
| `action_init_noise` | 0.048 | 0.050 | +0.002 | +4.38% |
| `attention_mask_to_device` | 0.006 | 0.007 | +0.000 | +5.82% |
| `embed_tokens` | 0.045 | 0.046 | +0.001 | +2.45% |
| `image_cast` | 0.047 | 0.048 | +0.001 | +2.00% |
| `kv_cache_trim` | 0.812 | 0.816 | +0.004 | +0.49% |
| `moe_indices` | 0.091 | 0.094 | +0.004 | +4.00% |
| `postfix_mask_build` | 0.131 | 0.132 | +0.001 | +0.87% |
| `postfix_moe_indices` | 0.103 | 0.106 | +0.003 | +3.31% |
| `postfix_slice` | 0.055 | 0.055 | +0.000 | +0.63% |
| `prefill_action_head` | 0.148 | 0.153 | +0.005 | +3.25% |
| `prefix_length_resolve` | 0.076 | 0.078 | +0.002 | +2.64% |
| `pruning_position_ids_prepare` | 0.556 | 0.568 | +0.012 | +2.19% |
| `scatter_action_init` | 0.055 | 0.057 | +0.002 | +3.72% |
| `scatter_proprioception` | 0.124 | 0.130 | +0.006 | +4.75% |
| `vispruner_apply_keep_to_sequences` | 0.192 | 0.197 | +0.004 | +2.20% |
| `vispruner_build_keep_mask` | 0.237 | 0.240 | +0.003 | +1.22% |
| `vispruner_gather_image_embeds` | 0.039 | 0.040 | +0.001 | +2.08% |
| `vispruner_image_lengths` | 0.063 | 0.064 | +0.001 | +1.05% |
| `vispruner_pad_pruned_batch` | 0.075 | 0.077 | +0.001 | +1.87% |
| `vispruner_rope_deltas` | 0.126 | 0.130 | +0.004 | +2.92% |
| `vispruner_score_prepare` | 0.030 | 0.031 | +0.000 | +0.84% |
| `vispruner_topk_select` | 0.133 | 0.134 | +0.002 | +1.17% |
| `vispruner_total` | 0.835 | 0.851 | +0.016 | +1.90% |

## Paired Samples

| idx | source | tokens | original_steps | modified_steps | original_total_ms | modified_total_ms | total_delta_pct | original_ode_ms | modified_ode_ms | ode_delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.202 | 335.054 | +0.26% | 276.408 | 277.855 | +0.52% |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.571 | 334.961 | +0.42% | 275.897 | 276.651 | +0.27% |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.539 | 337.019 | -0.74% | 280.872 | 279.156 | -0.61% |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 337.476 | 340.664 | +0.94% | 278.759 | 281.713 | +1.06% |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 337.303 | 336.186 | -0.33% | 278.577 | 278.070 | -0.18% |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.554 | 340.046 | +1.95% | 275.617 | 280.246 | +1.68% |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.040 | 335.662 | +0.79% | 275.201 | 277.964 | +1.00% |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.086 | 337.283 | +0.96% | 275.454 | 278.219 | +1.00% |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.734 | 338.858 | +1.84% | 274.884 | 280.871 | +2.18% |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.510 | 334.479 | +0.90% | 274.132 | 276.207 | +0.76% |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 330.580 | 335.011 | +1.34% | 273.075 | 275.700 | +0.96% |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 330.636 | 339.528 | +2.69% | 273.139 | 279.640 | +2.38% |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.869 | 336.963 | +1.53% | 273.779 | 277.597 | +1.39% |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.203 | 339.130 | +2.09% | 274.670 | 280.315 | +2.06% |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.656 | 337.810 | +1.86% | 274.354 | 277.739 | +1.23% |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.488 | 339.020 | +2.27% | 274.224 | 280.566 | +2.31% |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.112 | 335.211 | +0.63% | 274.446 | 276.080 | +0.60% |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.803 | 336.241 | +0.73% | 276.247 | 277.188 | +0.34% |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.549 | 335.778 | +0.67% | 275.703 | 277.476 | +0.64% |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 326.597 | 329.336 | +0.84% | 268.208 | 271.484 | +1.22% |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.569 | 336.632 | +1.53% | 274.072 | 277.537 | +1.26% |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.635 | 336.511 | +1.47% | 274.114 | 278.214 | +1.50% |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.254 | 337.687 | +1.33% | 275.376 | 279.152 | +1.37% |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.237 | 336.799 | +1.37% | 274.554 | 278.465 | +1.42% |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.530 | 335.744 | -1.11% | 280.333 | 277.120 | -1.15% |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 336.079 | 335.868 | -0.06% | 278.032 | 276.617 | -0.51% |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.849 | 334.826 | +0.29% | 275.902 | 276.635 | +0.27% |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.298 | 337.902 | +1.08% | 275.964 | 278.951 | +1.08% |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 329.908 | 327.552 | -0.71% | 272.594 | 269.931 | -0.98% |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 328.768 | 326.391 | -0.72% | 270.074 | 267.624 | -0.91% |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 330.882 | 327.539 | -1.01% | 273.228 | 270.026 | -1.17% |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 327.900 | 325.134 | -0.84% | 270.140 | 267.345 | -1.03% |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 330.317 | 334.004 | +1.12% | 273.051 | 276.141 | +1.13% |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.669 | 331.590 | -0.92% | 276.082 | 273.988 | -0.76% |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.252 | 333.196 | +0.28% | 274.736 | 275.334 | +0.22% |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.788 | 332.461 | +0.20% | 274.206 | 275.072 | +0.32% |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.091 | 333.795 | -0.09% | 275.476 | 276.197 | +0.26% |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.023 | 336.546 | +1.36% | 274.513 | 277.671 | +1.15% |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.078 | 336.622 | +1.06% | 275.548 | 277.761 | +0.80% |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.683 | 336.842 | +0.95% | 275.514 | 277.630 | +0.77% |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.887 | 334.970 | +0.32% | 275.655 | 277.114 | +0.53% |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 335.191 | 330.741 | -1.33% | 276.600 | 273.278 | -1.20% |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 331.409 | 331.350 | -0.02% | 273.959 | 273.928 | -0.01% |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.488 | 330.144 | -0.70% | 275.104 | 272.866 | -0.81% |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.600 | 332.350 | -0.08% | 274.505 | 273.568 | -0.34% |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 328.391 | 327.747 | -0.20% | 270.579 | 269.765 | -0.30% |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.136 | 340.203 | +1.82% | 276.521 | 281.602 | +1.84% |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 332.277 | 336.573 | +1.29% | 275.033 | 278.474 | +1.25% |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 334.827 | 334.120 | -0.21% | 277.494 | 276.009 | -0.54% |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 333.021 | 337.028 | +1.20% | 274.496 | 278.719 | +1.54% |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_original_vs_early_stop_results.json`