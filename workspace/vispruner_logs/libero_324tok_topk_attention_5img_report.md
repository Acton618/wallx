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
- pruned_strategy: `topk_attention`
- predictor_checkpoint: `None`
- predictor_source: `early_hidden`
- predictor_early_layer: `None`
- image_min_pixels: `254016`
- image_max_pixels: `None`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `5`
- tokens: baseline `324.00`, pruned `162.00`, delta `-162.00` (`-50.00%`)
- total_time: baseline `592.847 ms`, pruned `581.839 ms`, delta `-11.008 ms` (`-1.86%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 592.847 | 581.839 | -11.008 | -1.86% |
| `external_prepare_batch_ms` | 9.285 | 10.602 | +1.317 | +14.18% |
| `embed_processing` | 243.973 | 244.205 | +0.232 | +0.09% |
| `image_path_total` | 242.225 | 242.443 | +0.218 | +0.09% |
| `vision_image_forward` | 242.546 | 242.842 | +0.296 | +0.12% |
| `vision_image_encode` | 241.643 | 0.000 | -241.643 | -100.00% |
| `vision_image_encode_score` | 0.000 | 235.967 | +235.967 | +0.00% |
| `vispruner_total` | 0.000 | 3.884 | +3.884 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 1.233 | +1.233 | +0.00% |
| `vispruner_topk_select` | 0.000 | 0.484 | +0.484 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.991 | +0.991 | +0.00% |
| `position_encoding` | 2.235 | 0.374 | -1.861 | -83.27% |
| `prefetch_forward` | 41.566 | 33.564 | -8.002 | -19.25% |
| `prefill_transformer` | 40.076 | 32.339 | -7.737 | -19.31% |
| `cache_preprocessing` | 2.239 | 2.334 | +0.095 | +4.24% |
| `ode_integration` | 300.202 | 298.948 | -1.254 | -0.42% |
| `ode_transformer_total` | 271.597 | 271.998 | +0.402 | +0.15% |
| `postprocessing` | 0.003 | 0.002 | -0.001 | -22.33% |
| `action_init_embed` | 0.260 | 0.278 | +0.017 | +6.70% |
| `action_init_noise` | 0.154 | 0.113 | -0.041 | -26.37% |
| `action_initialization` | 1.346 | 1.160 | -0.186 | -13.84% |
| `attention_mask_to_device` | 0.002 | 0.002 | +0.000 | +15.46% |
| `embed_tokens` | 0.012 | 0.016 | +0.003 | +27.09% |
| `image_cast` | 0.019 | 0.015 | -0.005 | -24.28% |
| `kv_cache_trim` | 0.858 | 0.834 | -0.024 | -2.79% |
| `moe_indices` | 0.188 | 0.239 | +0.051 | +27.05% |
| `ode_action_embed_total` | 3.946 | 3.819 | -0.127 | -3.21% |
| `ode_action_head_total` | 2.188 | 2.028 | -0.160 | -7.32% |
| `ode_prepare_inputs` | 2.006 | 2.051 | +0.045 | +2.25% |
| `position_ids_rope` | 1.714 | 0.000 | -1.714 | -100.00% |
| `postfix_mask_build` | 0.389 | 0.525 | +0.136 | +34.96% |
| `postfix_moe_indices` | 0.313 | 0.370 | +0.058 | +18.42% |
| `postfix_slice` | 0.062 | 0.058 | -0.004 | -7.18% |
| `prefill_action_head` | 0.589 | 0.403 | -0.186 | -31.65% |
| `prefix_length_resolve` | 0.028 | 0.013 | -0.015 | -52.28% |
| `pruning_position_ids_prepare` | 0.000 | 1.569 | +1.569 | +0.00% |
| `scatter_action_init` | 0.197 | 0.156 | -0.041 | -20.91% |
| `scatter_image_embeds` | 0.272 | 0.198 | -0.074 | -27.09% |
| `scatter_proprioception` | 0.194 | 0.165 | -0.029 | -15.10% |
| `vispruner_gather_image_embeds` | 0.000 | 0.165 | +0.165 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.038 | +0.038 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.027 | +0.027 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.302 | +0.302 | +0.00% |
| `vispruner_score_prepare` | 0.000 | 0.145 | +0.145 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 595.994 | 581.025 | -14.969 | -2.51% |
| 2 | `dataset_index=67942 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 593.903 | 560.922 | -32.982 | -5.55% |
| 3 | `dataset_index=135885 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 596.767 | 578.882 | -17.886 | -3.00% |
| 4 | `dataset_index=203828 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 579.355 | 599.621 | +20.266 | +3.50% |
| 5 | `dataset_index=271771 image_key=observation.images.faceImg` | 324 | 162 | -50.00% | 598.218 | 588.748 | -9.470 | -1.58% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `workspace/vispruner_logs/libero_324tok_topk_attention_5img_results.json`