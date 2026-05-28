# Wall-X LeRobot Media VisPruner Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `40`
- num_videos: `0`
- video_frames_per_sample: `8`
- warmup: `1`
- iters: `3`
- keep_ratio: `0.5`
- pruned_strategy: `predictor_score`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `40`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `342.308 ms`, pruned `351.836 ms`, delta `+9.528 ms` (`+2.78%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 342.308 | 351.836 | +9.528 | +2.78% |
| `external_prepare_batch_ms` | 3.942 | 3.572 | -0.370 | -9.38% |
| `embed_processing` | 27.186 | 29.265 | +2.079 | +7.65% |
| `image_path_total` | 26.736 | 28.842 | +2.106 | +7.88% |
| `vision_image_forward` | 26.763 | 28.873 | +2.110 | +7.88% |
| `vision_image_encode` | 26.630 | 26.890 | +0.260 | +0.98% |
| `vispruner_total` | 0.000 | 0.880 | +0.880 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.256 | +0.256 | +0.00% |
| `vispruner_topk_select` | 0.000 | 0.140 | +0.140 | +0.00% |
| `vispruner_predictor_score` | 0.000 | 0.217 | +0.217 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.196 | +0.196 | +0.00% |
| `position_encoding` | 0.705 | 0.120 | -0.585 | -82.95% |
| `prefetch_forward` | 30.194 | 30.624 | +0.431 | +1.43% |
| `prefill_transformer` | 29.983 | 30.372 | +0.389 | +1.30% |
| `cache_preprocessing` | 1.316 | 1.348 | +0.032 | +2.45% |
| `ode_integration` | 282.299 | 289.842 | +7.542 | +2.67% |
| `ode_transformer_total` | 275.510 | 282.041 | +6.531 | +2.37% |
| `postprocessing` | 0.009 | 0.009 | -0.000 | -1.98% |
| `action_init_embed` | 0.281 | 0.306 | +0.025 | +8.90% |
| `action_init_noise` | 0.052 | 0.054 | +0.002 | +3.05% |
| `action_initialization` | 0.468 | 0.496 | +0.028 | +6.01% |
| `attention_mask_to_device` | 0.007 | 0.006 | -0.000 | -5.33% |
| `embed_tokens` | 0.050 | 0.051 | +0.001 | +2.46% |
| `image_cast` | 0.048 | 0.049 | +0.001 | +1.11% |
| `kv_cache_trim` | 0.833 | 0.839 | +0.006 | +0.75% |
| `moe_indices` | 0.096 | 0.099 | +0.004 | +3.87% |
| `ode_action_embed_total` | 2.822 | 3.241 | +0.420 | +14.87% |
| `ode_action_head_total` | 1.032 | 1.271 | +0.239 | +23.15% |
| `ode_prepare_inputs` | 0.663 | 0.730 | +0.066 | +9.96% |
| `position_ids_rope` | 0.564 | 0.000 | -0.564 | -100.00% |
| `postfix_mask_build` | 0.141 | 0.149 | +0.007 | +4.99% |
| `postfix_moe_indices` | 0.114 | 0.129 | +0.015 | +13.09% |
| `postfix_slice` | 0.057 | 0.056 | -0.001 | -1.29% |
| `prefill_action_head` | 0.153 | 0.188 | +0.034 | +22.34% |
| `prefix_length_resolve` | 0.083 | 0.089 | +0.006 | +7.11% |
| `pruning_position_ids_prepare` | 0.000 | 0.637 | +0.637 | +0.00% |
| `scatter_action_init` | 0.060 | 0.059 | -0.001 | -1.58% |
| `scatter_image_embeds` | 0.138 | 0.112 | -0.026 | -18.72% |
| `scatter_proprioception` | 0.136 | 0.137 | +0.000 | +0.11% |
| `vispruner_gather_image_embeds` | 0.000 | 0.041 | +0.041 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.068 | +0.068 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.078 | +0.078 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.137 | +0.137 | +0.00% |
| `vispruner_score_prepare` | 0.000 | 0.044 | +0.044 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 346.785 | 341.971 | -4.814 | -1.39% |
| 2 | `dataset_index=6968 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.132 | 339.299 | -10.832 | -3.09% |
| 3 | `dataset_index=13936 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 346.975 | 342.446 | -4.528 | -1.31% |
| 4 | `dataset_index=20905 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.786 | 344.658 | -6.127 | -1.75% |
| 5 | `dataset_index=27873 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 346.375 | 341.518 | -4.857 | -1.40% |
| 6 | `dataset_index=34842 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.771 | 338.916 | -4.854 | -1.41% |
| 7 | `dataset_index=41810 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 346.492 | 340.672 | -5.820 | -1.68% |
| 8 | `dataset_index=48779 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 339.856 | 337.354 | -2.502 | -0.74% |
| 9 | `dataset_index=55747 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 344.280 | 342.811 | -1.469 | -0.43% |
| 10 | `dataset_index=62716 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 342.026 | 338.639 | -3.387 | -0.99% |
| 11 | `dataset_index=69684 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 337.753 | 353.156 | +15.403 | +4.56% |
| 12 | `dataset_index=76653 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 342.741 | 357.033 | +14.292 | +4.17% |
| 13 | `dataset_index=83621 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.113 | 355.488 | +12.375 | +3.61% |
| 14 | `dataset_index=90590 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 340.167 | 358.915 | +18.748 | +5.51% |
| 15 | `dataset_index=97558 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 344.946 | 359.015 | +14.069 | +4.08% |
| 16 | `dataset_index=104527 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.204 | 362.265 | +19.061 | +5.55% |
| 17 | `dataset_index=111495 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 337.957 | 349.315 | +11.359 | +3.36% |
| 18 | `dataset_index=118464 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 355.103 | 358.506 | +3.403 | +0.96% |
| 19 | `dataset_index=125432 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 345.983 | 360.232 | +14.249 | +4.12% |
| 20 | `dataset_index=132401 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.449 | 371.990 | +28.541 | +8.31% |
| 21 | `dataset_index=139369 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 342.952 | 430.491 | +87.539 | +25.53% |
| 22 | `dataset_index=146338 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 339.508 | 406.775 | +67.267 | +19.81% |
| 23 | `dataset_index=153306 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 347.352 | 353.163 | +5.812 | +1.67% |
| 24 | `dataset_index=160275 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 339.483 | 351.077 | +11.594 | +3.42% |
| 25 | `dataset_index=167243 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 332.906 | 345.867 | +12.960 | +3.89% |
| 26 | `dataset_index=174212 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 337.425 | 345.348 | +7.922 | +2.35% |
| 27 | `dataset_index=181180 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.547 | 346.348 | +2.801 | +0.82% |
| 28 | `dataset_index=188149 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.119 | 340.955 | +6.836 | +2.05% |
| 29 | `dataset_index=195117 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 340.447 | 344.238 | +3.791 | +1.11% |
| 30 | `dataset_index=202086 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 346.379 | 345.469 | -0.910 | -0.26% |
| 31 | `dataset_index=209054 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.369 | 343.078 | +9.709 | +2.91% |
| 32 | `dataset_index=216023 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 336.947 | 346.477 | +9.530 | +2.83% |
| 33 | `dataset_index=222991 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 340.625 | 346.910 | +6.285 | +1.85% |
| 34 | `dataset_index=229960 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.221 | 348.320 | +5.099 | +1.49% |
| 35 | `dataset_index=236928 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.741 | 350.285 | +6.545 | +1.90% |
| 36 | `dataset_index=243897 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 340.539 | 352.644 | +12.105 | +3.55% |
| 37 | `dataset_index=250865 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 341.790 | 344.707 | +2.917 | +0.85% |
| 38 | `dataset_index=257834 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 338.646 | 348.648 | +10.002 | +2.95% |
| 39 | `dataset_index=264802 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 337.114 | 344.286 | +7.172 | +2.13% |
| 40 | `dataset_index=271771 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 340.303 | 344.148 | +3.845 | +1.13% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_predictor_score_40img_30k_l8_results.json`