# Wall-X LeRobot Media VisPruner Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `5`
- num_videos: `0`
- video_frames_per_sample: `8`
- warmup: `1`
- iters: `2`
- keep_ratio: `0.5`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- image_min_pixels: `254016`
- image_max_pixels: `None`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `5`
- tokens: baseline `324.00`, pruned `162.00`, delta `-162.00` (`-50.00%`)
- total_time: baseline `597.403 ms`, pruned `523.735 ms`, delta `-73.668 ms` (`-12.33%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 597.403 | 523.735 | -73.668 | -12.33% |
| `external_prepare_batch_ms` | 8.516 | 10.296 | +1.780 | +20.91% |
| `embed_processing` | 234.114 | 175.950 | -58.164 | -24.84% |
| `image_path_total` | 232.015 | 174.045 | -57.971 | -24.99% |
| `vision_image_forward` | 232.428 | 174.404 | -58.024 | -24.96% |
| `vision_image_encode` | 231.468 | 0.000 | -231.468 | -100.00% |
| `vispruner_total` | 0.000 | 2.995 | +2.995 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.419 | +0.419 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 168.239 | +168.239 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 1.246 | +1.246 | +0.00% |
| `position_encoding` | 2.370 | 0.365 | -2.005 | -84.59% |
| `prefetch_forward` | 40.463 | 32.954 | -7.509 | -18.56% |
| `prefill_transformer` | 38.741 | 31.714 | -7.027 | -18.14% |
| `cache_preprocessing` | 2.304 | 2.357 | +0.053 | +2.32% |
| `ode_integration` | 315.626 | 309.380 | -6.246 | -1.98% |
| `ode_transformer_total` | 286.997 | 281.418 | -5.579 | -1.94% |
| `postprocessing` | 0.003 | 0.002 | -0.001 | -18.52% |
| `action_init_embed` | 0.341 | 0.333 | -0.008 | -2.21% |
| `action_init_noise` | 0.115 | 0.181 | +0.066 | +57.48% |
| `action_initialization` | 1.312 | 1.432 | +0.120 | +9.18% |
| `attention_mask_to_device` | 0.003 | 0.002 | -0.001 | -25.60% |
| `embed_tokens` | 0.017 | 0.011 | -0.007 | -37.67% |
| `image_cast` | 0.018 | 0.015 | -0.003 | -16.47% |
| `kv_cache_trim` | 0.865 | 0.861 | -0.005 | -0.53% |
| `moe_indices` | 0.269 | 0.184 | -0.084 | -31.32% |
| `ode_action_embed_total` | 4.091 | 4.086 | -0.005 | -0.13% |
| `ode_action_head_total` | 2.244 | 1.956 | -0.288 | -12.82% |
| `ode_prepare_inputs` | 1.926 | 1.918 | -0.009 | -0.45% |
| `position_ids_rope` | 1.715 | 0.000 | -1.715 | -100.00% |
| `postfix_mask_build` | 0.509 | 0.446 | -0.063 | -12.33% |
| `postfix_moe_indices` | 0.337 | 0.341 | +0.003 | +0.97% |
| `postfix_slice` | 0.057 | 0.058 | +0.001 | +1.85% |
| `prefill_action_head` | 0.724 | 0.339 | -0.385 | -53.21% |
| `prefix_length_resolve` | 0.028 | 0.021 | -0.007 | -26.11% |
| `pruning_position_ids_prepare` | 0.000 | 1.671 | +1.671 | +0.00% |
| `scatter_action_init` | 0.155 | 0.200 | +0.045 | +29.30% |
| `scatter_image_embeds` | 0.250 | 0.200 | -0.050 | -20.11% |
| `scatter_proprioception` | 0.300 | 0.210 | -0.090 | -30.03% |
| `vispruner_gather_image_embeds` | 0.000 | 0.020 | +0.020 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.037 | +0.037 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.020 | +0.020 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.390 | +0.390 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 530.307 | 534.278 | +3.971 | +0.75% |
| 2 | `dataset_index=67942 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 585.641 | 533.168 | -52.473 | -8.96% |
| 3 | `dataset_index=135885 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 636.652 | 547.049 | -89.603 | -14.07% |
| 4 | `dataset_index=203828 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 611.495 | 532.180 | -79.315 | -12.97% |
| 5 | `dataset_index=271771 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 622.920 | 471.999 | -150.922 | -24.23% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `workspace/vispruner_logs/libero_324tok_predictor_early_l8_5img_results.json`