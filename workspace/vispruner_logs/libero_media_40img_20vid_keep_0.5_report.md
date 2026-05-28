# Wall-X LeRobot Media VisPruner Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `40`
- num_videos: `20`
- video_frames_per_sample: `8`
- warmup: `1`
- iters: `3`
- keep_ratio: `0.5`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `40`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `334.586 ms`, pruned `332.378 ms`, delta `-2.209 ms` (`-0.66%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 334.586 | 332.378 | -2.209 | -0.66% |
| `external_prepare_batch_ms` | 3.661 | 3.573 | -0.088 | -2.42% |
| `embed_processing` | 26.768 | 26.363 | -0.405 | -1.51% |
| `image_path_total` | 26.370 | 25.993 | -0.376 | -1.43% |
| `vision_image_forward` | 26.395 | 26.020 | -0.375 | -1.42% |
| `vision_image_encode` | 26.279 | 0.000 | -26.279 | -100.00% |
| `vision_image_encode_score` | 0.000 | 24.522 | +24.522 | +0.00% |
| `vispruner_total` | 0.000 | 0.804 | +0.804 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.223 | +0.223 | +0.00% |
| `vispruner_topk_select` | 0.000 | 0.127 | +0.127 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.188 | +0.188 | +0.00% |
| `position_encoding` | 0.620 | 0.108 | -0.512 | -82.58% |
| `prefetch_forward` | 29.392 | 29.092 | -0.300 | -1.02% |
| `prefill_transformer` | 29.200 | 28.898 | -0.302 | -1.03% |
| `cache_preprocessing` | 1.238 | 1.234 | -0.003 | -0.27% |
| `ode_integration` | 276.006 | 275.024 | -0.982 | -0.36% |
| `ode_transformer_total` | 269.724 | 268.754 | -0.970 | -0.36% |
| `postprocessing` | 0.009 | 0.008 | -0.000 | -2.41% |
| `action_init_embed` | 0.258 | 0.255 | -0.003 | -1.03% |
| `action_init_noise` | 0.045 | 0.043 | -0.001 | -2.58% |
| `action_initialization` | 0.428 | 0.424 | -0.004 | -0.97% |
| `attention_mask_to_device` | 0.006 | 0.006 | -0.000 | -3.27% |
| `embed_tokens` | 0.039 | 0.037 | -0.001 | -3.81% |
| `image_cast` | 0.043 | 0.040 | -0.003 | -6.27% |
| `kv_cache_trim` | 0.810 | 0.814 | +0.003 | +0.43% |
| `moe_indices` | 0.086 | 0.087 | +0.001 | +1.08% |
| `ode_action_embed_total` | 2.637 | 2.638 | +0.000 | +0.01% |
| `ode_action_head_total` | 0.939 | 0.939 | +0.000 | +0.02% |
| `ode_prepare_inputs` | 0.626 | 0.619 | -0.007 | -1.14% |
| `position_ids_rope` | 0.491 | 0.000 | -0.491 | -100.00% |
| `postfix_mask_build` | 0.122 | 0.119 | -0.003 | -2.34% |
| `postfix_moe_indices` | 0.099 | 0.098 | -0.001 | -0.93% |
| `postfix_slice` | 0.054 | 0.054 | -0.000 | -0.34% |
| `prefill_action_head` | 0.140 | 0.140 | +0.000 | +0.11% |
| `prefix_length_resolve` | 0.067 | 0.065 | -0.002 | -2.63% |
| `pruning_position_ids_prepare` | 0.000 | 0.504 | +0.504 | +0.00% |
| `scatter_action_init` | 0.056 | 0.055 | -0.001 | -1.61% |
| `scatter_image_embeds` | 0.116 | 0.096 | -0.020 | -16.92% |
| `scatter_proprioception` | 0.123 | 0.119 | -0.004 | -3.34% |
| `vispruner_gather_image_embeds` | 0.000 | 0.039 | +0.039 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.057 | +0.057 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.072 | +0.072 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.123 | +0.123 | +0.00% |
| `vispruner_score_prepare` | 0.000 | 0.026 | +0.026 | +0.00% |

## Video Samples

- samples: `20`
- tokens: baseline `324.00`, pruned `324.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `423.138 ms`, pruned `411.939 ms`, delta `-11.199 ms` (`-2.65%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 423.138 | 411.939 | -11.199 | -2.65% |
| `external_prepare_batch_ms` | 19.597 | 19.141 | -0.456 | -2.33% |
| `embed_processing` | 107.880 | 91.750 | -16.131 | -14.95% |
| `vision_video_forward` | 107.405 | 91.231 | -16.174 | -15.06% |
| `scatter_video_embeds` | 0.062 | 0.067 | +0.005 | +8.78% |
| `position_encoding` | 0.736 | 0.806 | +0.070 | +9.51% |
| `prefetch_forward` | 29.898 | 30.767 | +0.869 | +2.91% |
| `prefill_transformer` | 29.696 | 30.540 | +0.844 | +2.84% |
| `cache_preprocessing` | 1.267 | 1.339 | +0.073 | +5.73% |
| `ode_integration` | 282.757 | 286.635 | +3.878 | +1.37% |
| `ode_transformer_total` | 276.246 | 279.280 | +3.034 | +1.10% |
| `postprocessing` | 0.009 | 0.009 | +0.000 | +0.16% |
| `action_init_embed` | 0.280 | 0.311 | +0.031 | +11.21% |
| `action_init_noise` | 0.049 | 0.053 | +0.003 | +6.85% |
| `action_initialization` | 0.461 | 0.501 | +0.040 | +8.70% |
| `attention_mask_to_device` | 0.007 | 0.007 | +0.000 | +2.13% |
| `embed_tokens` | 0.048 | 0.052 | +0.004 | +8.09% |
| `kv_cache_trim` | 0.812 | 0.842 | +0.030 | +3.69% |
| `moe_indices` | 0.093 | 0.101 | +0.009 | +9.47% |
| `ode_action_embed_total` | 2.723 | 3.074 | +0.352 | +12.91% |
| `ode_action_head_total` | 0.975 | 1.180 | +0.205 | +21.05% |
| `ode_prepare_inputs` | 0.638 | 0.687 | +0.048 | +7.60% |
| `position_ids_rope` | 0.598 | 0.658 | +0.060 | +9.99% |
| `postfix_mask_build` | 0.133 | 0.146 | +0.013 | +10.08% |
| `postfix_moe_indices` | 0.105 | 0.124 | +0.018 | +17.45% |
| `postfix_slice` | 0.055 | 0.057 | +0.003 | +4.96% |
| `prefill_action_head` | 0.153 | 0.176 | +0.023 | +15.11% |
| `prefix_length_resolve` | 0.078 | 0.085 | +0.007 | +8.36% |
| `scatter_action_init` | 0.058 | 0.060 | +0.002 | +3.24% |
| `scatter_proprioception` | 0.141 | 0.159 | +0.019 | +13.35% |

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 335.957 | 334.273 | -1.684 | -0.50% |
| 2 | `dataset_index=6968 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 344.540 | 333.626 | -10.915 | -3.17% |
| 3 | `dataset_index=13936 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 335.420 | 331.246 | -4.174 | -1.24% |
| 4 | `dataset_index=20905 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 335.347 | 335.975 | +0.627 | +0.19% |
| 5 | `dataset_index=27873 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 336.581 | 340.271 | +3.690 | +1.10% |
| 6 | `dataset_index=34842 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.330 | 330.037 | -3.294 | -0.99% |
| 7 | `dataset_index=41810 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.802 | 330.066 | -4.735 | -1.41% |
| 8 | `dataset_index=48779 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.800 | 331.694 | -2.106 | -0.63% |
| 9 | `dataset_index=55747 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 348.695 | 329.452 | -19.243 | -5.52% |
| 10 | `dataset_index=62716 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 341.737 | 329.653 | -12.084 | -3.54% |
| 11 | `dataset_index=69684 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.414 | 325.299 | -8.115 | -2.43% |
| 12 | `dataset_index=76653 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 338.473 | 337.100 | -1.373 | -0.41% |
| 13 | `dataset_index=83621 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 331.819 | 330.963 | -0.856 | -0.26% |
| 14 | `dataset_index=90590 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 335.818 | 331.364 | -4.454 | -1.33% |
| 15 | `dataset_index=97558 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.570 | 338.405 | +3.835 | +1.15% |
| 16 | `dataset_index=104527 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 331.151 | 333.752 | +2.601 | +0.79% |
| 17 | `dataset_index=111495 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 329.565 | 333.070 | +3.506 | +1.06% |
| 18 | `dataset_index=118464 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 329.736 | 333.826 | +4.091 | +1.24% |
| 19 | `dataset_index=125432 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 332.885 | 335.020 | +2.136 | +0.64% |
| 20 | `dataset_index=132401 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 332.585 | 335.300 | +2.715 | +0.82% |
| 21 | `dataset_index=139369 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 332.771 | 330.489 | -2.282 | -0.69% |
| 22 | `dataset_index=146338 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.584 | 331.801 | -2.782 | -0.83% |
| 23 | `dataset_index=153306 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.776 | 330.503 | -4.273 | -1.28% |
| 24 | `dataset_index=160275 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.480 | 331.853 | -1.627 | -0.49% |
| 25 | `dataset_index=167243 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 329.813 | 325.037 | -4.775 | -1.45% |
| 26 | `dataset_index=174212 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 332.115 | 330.493 | -1.622 | -0.49% |
| 27 | `dataset_index=181180 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.151 | 334.412 | +1.261 | +0.38% |
| 28 | `dataset_index=188149 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 328.362 | 327.436 | -0.926 | -0.28% |
| 29 | `dataset_index=195117 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 332.145 | 333.647 | +1.502 | +0.45% |
| 30 | `dataset_index=202086 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.638 | 348.507 | +13.870 | +4.14% |
| 31 | `dataset_index=209054 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 328.649 | 323.644 | -5.005 | -1.52% |
| 32 | `dataset_index=216023 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 329.399 | 331.900 | +2.500 | +0.76% |
| 33 | `dataset_index=222991 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 330.979 | 329.830 | -1.149 | -0.35% |
| 34 | `dataset_index=229960 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 329.964 | 328.968 | -0.996 | -0.30% |
| 35 | `dataset_index=236928 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 351.754 | 328.535 | -23.219 | -6.60% |
| 36 | `dataset_index=243897 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 335.717 | 332.901 | -2.816 | -0.84% |
| 37 | `dataset_index=250865 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 337.530 | 333.693 | -3.837 | -1.14% |
| 38 | `dataset_index=257834 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 335.675 | 333.600 | -2.076 | -0.62% |
| 39 | `dataset_index=264802 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 333.358 | 334.838 | +1.479 | +0.44% |
| 40 | `dataset_index=271771 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 334.370 | 332.618 | -1.752 | -0.52% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `episode_000000.mp4` | 324 | 324 | +0.00% | 416.131 | 410.703 | -5.428 | -1.30% |
| 2 | `episode_000001.mp4` | 324 | 324 | +0.00% | 419.307 | 403.206 | -16.101 | -3.84% |
| 3 | `episode_000002.mp4` | 324 | 324 | +0.00% | 427.495 | 412.213 | -15.282 | -3.57% |
| 4 | `episode_000003.mp4` | 324 | 324 | +0.00% | 428.389 | 415.307 | -13.082 | -3.05% |
| 5 | `episode_000004.mp4` | 324 | 324 | +0.00% | 418.656 | 412.776 | -5.880 | -1.40% |
| 6 | `episode_000005.mp4` | 324 | 324 | +0.00% | 434.178 | 417.201 | -16.977 | -3.91% |
| 7 | `episode_000006.mp4` | 324 | 324 | +0.00% | 417.537 | 415.829 | -1.708 | -0.41% |
| 8 | `episode_000007.mp4` | 324 | 324 | +0.00% | 422.269 | 414.401 | -7.868 | -1.86% |
| 9 | `episode_000008.mp4` | 324 | 324 | +0.00% | 426.706 | 412.340 | -14.366 | -3.37% |
| 10 | `episode_000009.mp4` | 324 | 324 | +0.00% | 455.490 | 409.514 | -45.976 | -10.09% |
| 11 | `episode_000010.mp4` | 324 | 324 | +0.00% | 417.260 | 408.041 | -9.219 | -2.21% |
| 12 | `episode_000011.mp4` | 324 | 324 | +0.00% | 415.233 | 410.816 | -4.416 | -1.06% |
| 13 | `episode_000012.mp4` | 324 | 324 | +0.00% | 414.014 | 410.878 | -3.136 | -0.76% |
| 14 | `episode_000013.mp4` | 324 | 324 | +0.00% | 415.582 | 408.195 | -7.387 | -1.78% |
| 15 | `episode_000014.mp4` | 324 | 324 | +0.00% | 418.712 | 409.343 | -9.369 | -2.24% |
| 16 | `episode_000015.mp4` | 324 | 324 | +0.00% | 423.647 | 410.613 | -13.034 | -3.08% |
| 17 | `episode_000016.mp4` | 324 | 324 | +0.00% | 421.311 | 412.975 | -8.336 | -1.98% |
| 18 | `episode_000017.mp4` | 324 | 324 | +0.00% | 426.928 | 420.819 | -6.109 | -1.43% |
| 19 | `episode_000018.mp4` | 324 | 324 | +0.00% | 421.265 | 411.610 | -9.655 | -2.29% |
| 20 | `episode_000019.mp4` | 324 | 324 | +0.00% | 422.653 | 412.001 | -10.653 | -2.52% |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_media_40img_20vid_keep_0.5_results.json`