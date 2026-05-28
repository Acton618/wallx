# Wall-X LeRobot Media VisPruner Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `1`
- num_videos: `0`
- video_frames_per_sample: `8`
- warmup: `0`
- iters: `1`
- keep_ratio: `0.5`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `1`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `571.201 ms`, pruned `395.372 ms`, delta `-175.829 ms` (`-30.78%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 571.201 | 395.372 | -175.829 | -30.78% |
| `external_prepare_batch_ms` | 9.007 | 4.911 | -4.095 | -45.47% |
| `embed_processing` | 232.681 | 33.566 | -199.116 | -85.57% |
| `image_path_total` | 229.653 | 33.040 | -196.613 | -85.61% |
| `vision_image_forward` | 229.684 | 33.073 | -196.611 | -85.60% |
| `vision_image_encode` | 229.521 | 0.000 | -229.521 | -100.00% |
| `vispruner_total` | 0.000 | 0.737 | +0.737 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.056 | +0.056 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 31.422 | +31.422 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.213 | +0.213 | +0.00% |
| `position_encoding` | 1.008 | 0.126 | -0.882 | -87.49% |
| `prefetch_forward` | 39.396 | 31.556 | -7.841 | -19.90% |
| `prefill_transformer` | 39.096 | 31.320 | -7.776 | -19.89% |
| `cache_preprocessing` | 3.390 | 1.358 | -2.033 | -59.95% |
| `ode_integration` | 291.558 | 327.953 | +36.395 | +12.48% |
| `ode_transformer_total` | 283.370 | 318.912 | +35.542 | +12.54% |
| `postprocessing` | 0.009 | 0.011 | +0.002 | +25.71% |
| `action_init_embed` | 1.970 | 0.411 | -1.560 | -79.16% |
| `action_init_noise` | 0.674 | 0.096 | -0.578 | -85.82% |
| `action_initialization` | 2.851 | 0.653 | -2.198 | -77.11% |
| `attention_mask_to_device` | 0.015 | 0.008 | -0.007 | -48.65% |
| `embed_tokens` | 0.358 | 0.077 | -0.282 | -78.57% |
| `image_cast` | 0.060 | 0.053 | -0.008 | -12.49% |
| `kv_cache_trim` | 0.863 | 0.837 | -0.026 | -3.03% |
| `moe_indices` | 0.147 | 0.104 | -0.043 | -29.36% |
| `ode_action_embed_total` | 2.994 | 3.760 | +0.765 | +25.56% |
| `ode_action_head_total` | 1.046 | 1.475 | +0.429 | +40.97% |
| `ode_prepare_inputs` | 0.687 | 0.877 | +0.190 | +27.72% |
| `position_ids_rope` | 0.803 | 0.000 | -0.803 | -100.00% |
| `postfix_mask_build` | 0.327 | 0.152 | -0.175 | -53.56% |
| `postfix_moe_indices` | 0.135 | 0.114 | -0.021 | -15.70% |
| `postfix_slice` | 0.057 | 0.056 | -0.001 | -1.24% |
| `prefill_action_head` | 0.240 | 0.174 | -0.066 | -27.63% |
| `prefix_length_resolve` | 1.902 | 0.108 | -1.794 | -94.31% |
| `pruning_position_ids_prepare` | 0.000 | 0.665 | +0.665 | +0.00% |
| `scatter_action_init` | 0.084 | 0.063 | -0.022 | -25.57% |
| `scatter_image_embeds` | 0.266 | 0.137 | -0.129 | -48.40% |
| `scatter_proprioception` | 2.243 | 0.180 | -2.062 | -91.95% |
| `vispruner_gather_image_embeds` | 0.000 | 0.031 | +0.031 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.099 | +0.099 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.085 | +0.085 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.145 | +0.145 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 571.201 | 395.372 | -175.829 | -30.78% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `workspace/vispruner_logs/libero_predictor_early_1img_results.json`