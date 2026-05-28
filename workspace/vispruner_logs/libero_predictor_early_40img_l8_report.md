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
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `40`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `351.514 ms`, pruned `357.677 ms`, delta `+6.163 ms` (`+1.75%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 351.514 | 357.677 | +6.163 | +1.75% |
| `external_prepare_batch_ms` | 3.655 | 3.931 | +0.276 | +7.54% |
| `embed_processing` | 28.599 | 31.321 | +2.721 | +9.52% |
| `image_path_total` | 28.119 | 30.890 | +2.771 | +9.85% |
| `vision_image_forward` | 28.147 | 30.920 | +2.773 | +9.85% |
| `vision_image_encode` | 28.012 | 0.000 | -28.012 | -100.00% |
| `vispruner_total` | 0.000 | 0.709 | +0.709 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.053 | +0.053 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 29.344 | +29.344 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.210 | +0.210 | +0.00% |
| `position_encoding` | 0.728 | 0.121 | -0.607 | -83.42% |
| `prefetch_forward` | 30.971 | 30.904 | -0.067 | -0.22% |
| `prefill_transformer` | 30.739 | 30.640 | -0.099 | -0.32% |
| `cache_preprocessing` | 1.358 | 1.388 | +0.030 | +2.20% |
| `ode_integration` | 289.228 | 293.299 | +4.071 | +1.41% |
| `ode_transformer_total` | 281.953 | 285.173 | +3.220 | +1.14% |
| `postprocessing` | 0.009 | 0.009 | -0.000 | -0.48% |
| `action_init_embed` | 0.292 | 0.308 | +0.016 | +5.37% |
| `action_init_noise` | 0.054 | 0.053 | -0.001 | -1.60% |
| `action_initialization` | 0.485 | 0.500 | +0.016 | +3.25% |
| `attention_mask_to_device` | 0.007 | 0.007 | -0.000 | -6.88% |
| `embed_tokens` | 0.053 | 0.053 | -0.000 | -0.02% |
| `image_cast` | 0.051 | 0.050 | -0.001 | -1.32% |
| `kv_cache_trim` | 0.854 | 0.866 | +0.011 | +1.34% |
| `moe_indices` | 0.099 | 0.099 | +0.001 | +0.57% |
| `ode_action_embed_total` | 2.995 | 3.344 | +0.349 | +11.64% |
| `ode_action_head_total` | 1.141 | 1.349 | +0.208 | +18.20% |
| `ode_prepare_inputs` | 0.710 | 0.756 | +0.046 | +6.46% |
| `position_ids_rope` | 0.582 | 0.000 | -0.582 | -100.00% |
| `postfix_mask_build` | 0.148 | 0.153 | +0.005 | +3.12% |
| `postfix_moe_indices` | 0.119 | 0.130 | +0.010 | +8.72% |
| `postfix_slice` | 0.058 | 0.059 | +0.001 | +2.21% |
| `prefill_action_head` | 0.169 | 0.195 | +0.026 | +15.39% |
| `prefix_length_resolve` | 0.087 | 0.090 | +0.003 | +3.71% |
| `pruning_position_ids_prepare` | 0.000 | 0.637 | +0.637 | +0.00% |
| `scatter_action_init` | 0.062 | 0.061 | -0.001 | -1.61% |
| `scatter_image_embeds` | 0.150 | 0.113 | -0.037 | -24.70% |
| `scatter_proprioception` | 0.145 | 0.140 | -0.005 | -3.49% |
| `vispruner_gather_image_embeds` | 0.000 | 0.030 | +0.030 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.085 | +0.085 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.084 | +0.084 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.138 | +0.138 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.501 | 354.764 | +4.263 | +1.22% |
| 2 | `dataset_index=6968 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 354.751 | 357.861 | +3.110 | +0.88% |
| 3 | `dataset_index=13936 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 352.605 | 360.373 | +7.768 | +2.20% |
| 4 | `dataset_index=20905 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 354.057 | 370.757 | +16.700 | +4.72% |
| 5 | `dataset_index=27873 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 347.619 | 356.577 | +8.958 | +2.58% |
| 6 | `dataset_index=34842 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.270 | 377.663 | +27.393 | +7.82% |
| 7 | `dataset_index=41810 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 352.800 | 358.813 | +6.013 | +1.70% |
| 8 | `dataset_index=48779 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.740 | 359.737 | +8.997 | +2.57% |
| 9 | `dataset_index=55747 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 354.029 | 360.442 | +6.414 | +1.81% |
| 10 | `dataset_index=62716 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 349.540 | 375.311 | +25.771 | +7.37% |
| 11 | `dataset_index=69684 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 342.546 | 351.869 | +9.323 | +2.72% |
| 12 | `dataset_index=76653 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 351.158 | 356.582 | +5.424 | +1.54% |
| 13 | `dataset_index=83621 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 349.421 | 355.874 | +6.453 | +1.85% |
| 14 | `dataset_index=90590 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.382 | 355.945 | +5.563 | +1.59% |
| 15 | `dataset_index=97558 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 350.830 | 350.647 | -0.183 | -0.05% |
| 16 | `dataset_index=104527 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 348.931 | 355.792 | +6.861 | +1.97% |
| 17 | `dataset_index=111495 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 344.908 | 353.368 | +8.459 | +2.45% |
| 18 | `dataset_index=118464 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 344.057 | 356.645 | +12.588 | +3.66% |
| 19 | `dataset_index=125432 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 348.418 | 362.781 | +14.363 | +4.12% |
| 20 | `dataset_index=132401 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 343.262 | 375.035 | +31.773 | +9.26% |
| 21 | `dataset_index=139369 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 348.958 | 359.267 | +10.308 | +2.95% |
| 22 | `dataset_index=146338 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 348.328 | 372.022 | +23.695 | +6.80% |
| 23 | `dataset_index=153306 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 344.846 | 360.334 | +15.488 | +4.49% |
| 24 | `dataset_index=160275 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 349.331 | 360.170 | +10.839 | +3.10% |
| 25 | `dataset_index=167243 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 337.115 | 349.945 | +12.830 | +3.81% |
| 26 | `dataset_index=174212 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 349.394 | 354.500 | +5.106 | +1.46% |
| 27 | `dataset_index=181180 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 346.662 | 354.440 | +7.778 | +2.24% |
| 28 | `dataset_index=188149 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 348.880 | 345.251 | -3.630 | -1.04% |
| 29 | `dataset_index=195117 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 359.967 | 352.067 | -7.900 | -2.19% |
| 30 | `dataset_index=202086 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 358.888 | 347.093 | -11.795 | -3.29% |
| 31 | `dataset_index=209054 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 367.978 | 341.422 | -26.556 | -7.22% |
| 32 | `dataset_index=216023 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 373.405 | 353.864 | -19.541 | -5.23% |
| 33 | `dataset_index=222991 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 351.270 | 356.886 | +5.616 | +1.60% |
| 34 | `dataset_index=229960 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 352.861 | 355.721 | +2.860 | +0.81% |
| 35 | `dataset_index=236928 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 353.535 | 357.162 | +3.627 | +1.03% |
| 36 | `dataset_index=243897 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 354.000 | 358.488 | +4.488 | +1.27% |
| 37 | `dataset_index=250865 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 354.013 | 355.340 | +1.327 | +0.37% |
| 38 | `dataset_index=257834 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 354.174 | 356.115 | +1.941 | +0.55% |
| 39 | `dataset_index=264802 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 357.511 | 354.998 | -2.513 | -0.70% |
| 40 | `dataset_index=271771 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 358.610 | 355.158 | -3.452 | -0.96% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `workspace/vispruner_logs/libero_predictor_early_40img_l8_results.json`