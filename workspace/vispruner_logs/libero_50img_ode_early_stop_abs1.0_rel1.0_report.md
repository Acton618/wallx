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
- early_stop_abs_threshold: `1.0`
- early_stop_rel_threshold: `1.0`
- warmup: `1`
- iters: `3`
- device: `cuda`

## Summary

- samples: `50`
- tokens: `81.00` -> `41.00`
- original total_time: `378.563 ms`
- modified total_time: `101.209 ms`
- total_time delta: `-277.354 ms` (`-73.26%`)
- original ode_integration: `312.406 ms`
- modified ode_integration: `35.041 ms`
- ode_integration delta: `-277.365 ms` (`-88.78%`)
- original ode_steps_used: `9.00`
- modified ode_steps_used: `1.00`
- action MAE vs original: `2.078964`
- action RMSE vs original: `2.663640`
- action max_abs vs original: `5.992279`

## Stage Timing

| stage | original_ms | modified_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 378.563 | 101.209 | -277.354 | -73.26% |
| `external_prepare_batch_ms` | 3.980 | 4.031 | +0.051 | +1.28% |
| `embed_processing` | 30.772 | 30.718 | -0.055 | -0.18% |
| `image_path_total` | 30.277 | 30.245 | -0.033 | -0.11% |
| `vision_image_forward` | 30.310 | 30.276 | -0.034 | -0.11% |
| `vision_image_encode_score` | 28.344 | 28.357 | +0.014 | +0.05% |
| `scatter_image_embeds` | 0.134 | 0.129 | -0.005 | -3.98% |
| `position_encoding` | 0.136 | 0.128 | -0.008 | -5.56% |
| `action_initialization` | 0.567 | 0.544 | -0.023 | -3.97% |
| `prefetch_forward` | 33.054 | 33.133 | +0.079 | +0.24% |
| `prefill_transformer` | 32.769 | 32.852 | +0.083 | +0.25% |
| `cache_preprocessing` | 1.475 | 1.496 | +0.021 | +1.41% |
| `ode_integration` | 312.406 | 35.041 | -277.365 | -88.78% |
| `ode_action_embed_total` | 3.573 | 0.392 | -3.181 | -89.02% |
| `ode_prepare_inputs` | 0.806 | 0.087 | -0.719 | -89.20% |
| `ode_transformer_total` | 303.673 | 34.114 | -269.558 | -88.77% |
| `ode_action_head_total` | 1.468 | 0.158 | -1.310 | -89.24% |
| `postprocessing` | 0.007 | 0.007 | -0.000 | -3.89% |
| `action_init_embed` | 0.346 | 0.334 | -0.012 | -3.55% |
| `action_init_noise` | 0.064 | 0.060 | -0.004 | -6.02% |
| `attention_mask_to_device` | 0.008 | 0.008 | -0.000 | -4.06% |
| `embed_tokens` | 0.059 | 0.057 | -0.003 | -4.81% |
| `image_cast` | 0.057 | 0.054 | -0.003 | -6.03% |
| `kv_cache_trim` | 0.914 | 0.935 | +0.021 | +2.29% |
| `moe_indices` | 0.112 | 0.105 | -0.006 | -5.78% |
| `postfix_mask_build` | 0.163 | 0.161 | -0.002 | -1.02% |
| `postfix_moe_indices` | 0.141 | 0.141 | -0.000 | -0.08% |
| `postfix_slice` | 0.063 | 0.064 | +0.001 | +1.96% |
| `prefill_action_head` | 0.215 | 0.211 | -0.004 | -1.76% |
| `prefix_length_resolve` | 0.103 | 0.100 | -0.002 | -2.34% |
| `pruning_position_ids_prepare` | 0.713 | 0.688 | -0.024 | -3.42% |
| `scatter_action_init` | 0.067 | 0.066 | -0.002 | -2.56% |
| `scatter_proprioception` | 0.162 | 0.154 | -0.008 | -5.01% |
| `vispruner_apply_keep_to_sequences` | 0.230 | 0.230 | -0.000 | -0.06% |
| `vispruner_build_keep_mask` | 0.287 | 0.279 | -0.008 | -2.79% |
| `vispruner_gather_image_embeds` | 0.047 | 0.047 | -0.000 | -0.06% |
| `vispruner_image_lengths` | 0.078 | 0.077 | -0.001 | -1.06% |
| `vispruner_pad_pruned_batch` | 0.090 | 0.088 | -0.002 | -2.46% |
| `vispruner_rope_deltas` | 0.158 | 0.157 | -0.001 | -0.62% |
| `vispruner_score_prepare` | 0.037 | 0.036 | -0.001 | -2.83% |
| `vispruner_topk_select` | 0.164 | 0.160 | -0.004 | -2.38% |
| `vispruner_total` | 1.003 | 0.990 | -0.013 | -1.31% |

## Paired Samples

| idx | source | tokens | original_steps | modified_steps | original_total_ms | modified_total_ms | total_delta_pct | original_ode_ms | modified_ode_ms | ode_delta_pct | action_mae | action_rmse | action_max_abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.851 | 101.356 | -73.18% | 311.939 | 34.862 | -88.82% | 2.090074 | 2.677088 | 6.058411 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.673 | 102.933 | -72.67% | 311.340 | 36.576 | -88.25% | 2.085507 | 2.667438 | 6.035292 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.859 | 103.324 | -72.66% | 310.898 | 36.146 | -88.37% | 2.078386 | 2.655328 | 5.928982 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 371.964 | 102.748 | -72.38% | 307.230 | 36.163 | -88.23% | 2.079555 | 2.659445 | 6.049017 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 373.428 | 100.232 | -73.16% | 308.600 | 34.834 | -88.71% | 2.082165 | 2.672003 | 6.062614 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 379.931 | 98.751 | -74.01% | 313.722 | 33.541 | -89.31% | 2.080077 | 2.658972 | 5.955895 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 384.495 | 97.888 | -74.54% | 316.774 | 33.741 | -89.35% | 2.080801 | 2.665398 | 6.010708 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 384.749 | 100.771 | -73.81% | 317.750 | 35.539 | -88.82% | 2.080950 | 2.666418 | 6.007959 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 383.662 | 100.653 | -73.77% | 316.821 | 34.272 | -89.18% | 2.092336 | 2.674617 | 6.018600 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.096 | 101.733 | -73.02% | 311.130 | 35.144 | -88.70% | 2.077817 | 2.661711 | 5.988896 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 374.579 | 101.385 | -72.93% | 309.454 | 35.377 | -88.57% | 2.074614 | 2.659333 | 5.999165 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 378.629 | 101.299 | -73.25% | 312.532 | 35.873 | -88.52% | 2.073275 | 2.655829 | 6.052191 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 380.808 | 103.749 | -72.76% | 314.288 | 35.322 | -88.76% | 2.078982 | 2.662817 | 5.939008 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 378.403 | 101.533 | -73.17% | 313.194 | 34.519 | -88.98% | 2.076486 | 2.665346 | 6.031652 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 375.280 | 99.496 | -73.49% | 310.352 | 34.443 | -88.90% | 2.077295 | 2.660268 | 6.062253 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 382.334 | 98.729 | -74.18% | 316.027 | 35.361 | -88.81% | 2.077636 | 2.660310 | 5.905472 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.724 | 103.044 | -72.65% | 310.998 | 35.507 | -88.58% | 2.083221 | 2.671914 | 5.883391 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 378.635 | 100.537 | -73.45% | 312.743 | 34.581 | -88.94% | 2.075576 | 2.659683 | 6.039970 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 387.273 | 101.437 | -73.81% | 318.731 | 35.372 | -88.90% | 2.073484 | 2.660998 | 5.961189 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 379.948 | 100.333 | -73.59% | 312.588 | 34.498 | -88.96% | 2.074842 | 2.660910 | 5.966830 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.468 | 101.891 | -73.01% | 311.785 | 35.629 | -88.57% | 2.081689 | 2.663547 | 6.023075 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.543 | 100.208 | -73.39% | 310.515 | 34.952 | -88.74% | 2.080349 | 2.665605 | 6.077627 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 374.033 | 100.460 | -73.14% | 308.966 | 35.109 | -88.64% | 2.072103 | 2.656711 | 5.928911 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 381.127 | 98.809 | -74.07% | 314.681 | 33.953 | -89.21% | 2.079082 | 2.671664 | 5.992006 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 386.805 | 103.989 | -73.12% | 320.116 | 37.040 | -88.43% | 2.080148 | 2.660059 | 5.906504 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 387.683 | 102.631 | -73.53% | 321.560 | 35.003 | -89.11% | 2.071992 | 2.652157 | 5.976177 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 374.784 | 100.977 | -73.06% | 309.102 | 34.720 | -88.77% | 2.075071 | 2.661083 | 5.944573 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 374.159 | 100.203 | -73.22% | 307.847 | 34.964 | -88.64% | 2.074915 | 2.654074 | 6.005266 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.653 | 101.087 | -73.16% | 309.367 | 34.256 | -88.93% | 2.067622 | 2.649999 | 5.915047 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 396.410 | 102.907 | -74.04% | 326.951 | 35.368 | -89.18% | 2.073406 | 2.656599 | 5.997302 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 380.270 | 101.717 | -73.25% | 312.923 | 34.913 | -88.84% | 2.073224 | 2.650802 | 5.945803 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 372.017 | 100.964 | -72.86% | 305.158 | 34.654 | -88.64% | 2.078514 | 2.663961 | 5.934257 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 369.866 | 101.415 | -72.58% | 305.578 | 34.827 | -88.60% | 2.081749 | 2.670104 | 6.002073 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 371.220 | 102.606 | -72.36% | 305.592 | 35.184 | -88.49% | 2.072150 | 2.662027 | 5.947947 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.224 | 101.054 | -73.21% | 311.261 | 35.309 | -88.66% | 2.072960 | 2.660428 | 6.062059 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 380.025 | 98.923 | -73.97% | 314.028 | 34.225 | -89.10% | 2.091737 | 2.675804 | 6.020697 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 375.291 | 100.920 | -73.11% | 309.666 | 35.213 | -88.63% | 2.081170 | 2.663368 | 6.004152 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 379.668 | 101.504 | -73.27% | 314.224 | 34.809 | -88.92% | 2.083807 | 2.669506 | 6.064526 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 372.907 | 103.069 | -72.36% | 308.064 | 35.809 | -88.38% | 2.085613 | 2.663216 | 5.988098 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 382.866 | 100.922 | -73.64% | 315.761 | 34.578 | -89.05% | 2.081885 | 2.668787 | 6.104830 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 384.691 | 102.846 | -73.27% | 317.142 | 35.301 | -88.87% | 2.089246 | 2.674613 | 6.043674 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.815 | 100.416 | -73.42% | 312.590 | 34.852 | -88.85% | 2.078373 | 2.665605 | 5.932873 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 378.494 | 100.420 | -73.47% | 312.989 | 34.092 | -89.11% | 2.086543 | 2.675450 | 5.979476 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.773 | 100.215 | -73.40% | 310.584 | 34.418 | -88.92% | 2.080354 | 2.668244 | 5.983805 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.268 | 102.125 | -72.86% | 310.665 | 36.292 | -88.32% | 2.082190 | 2.669688 | 6.043189 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 375.311 | 100.979 | -73.09% | 308.441 | 34.312 | -88.88% | 2.075258 | 2.661079 | 5.892851 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 376.277 | 102.935 | -72.64% | 311.002 | 35.332 | -88.64% | 2.079281 | 2.665730 | 5.953265 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 379.542 | 101.913 | -73.15% | 313.465 | 35.039 | -88.82% | 2.071887 | 2.656799 | 5.918229 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.848 | 100.829 | -73.31% | 311.807 | 35.256 | -88.69% | 2.075229 | 2.664530 | 6.049825 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 81->41 | 9.00 | 1.00 | 377.787 | 99.597 | -73.64% | 311.373 | 34.975 | -88.77% | 2.077579 | 2.664949 | 6.018364 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_early_stop_abs1.0_rel1.0_results.json`