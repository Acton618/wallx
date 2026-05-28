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
- pruned_strategy: `predictor_score`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_smoke.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `None`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `1`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `593.961 ms`, pruned `356.480 ms`, delta `-237.482 ms` (`-39.98%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 593.961 | 356.480 | -237.482 | -39.98% |
| `external_prepare_batch_ms` | 8.434 | 4.122 | -4.312 | -51.13% |
| `embed_processing` | 230.038 | 29.537 | -200.501 | -87.16% |
| `image_path_total` | 226.550 | 29.055 | -197.496 | -87.18% |
| `vision_image_forward` | 226.582 | 29.086 | -197.496 | -87.16% |
| `vision_image_encode` | 226.423 | 27.197 | -199.226 | -87.99% |
| `vispruner_total` | 0.000 | 0.879 | +0.879 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.251 | +0.251 | +0.00% |
| `vispruner_topk_select` | 0.000 | 0.136 | +0.136 | +0.00% |
| `vispruner_predictor_score` | 0.000 | 0.198 | +0.198 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.201 | +0.201 | +0.00% |
| `position_encoding` | 1.041 | 0.123 | -0.918 | -88.20% |
| `prefetch_forward` | 44.667 | 31.169 | -13.498 | -30.22% |
| `prefill_transformer` | 44.331 | 30.931 | -13.400 | -30.23% |
| `cache_preprocessing` | 6.524 | 1.367 | -5.157 | -79.05% |
| `ode_integration` | 308.524 | 293.558 | -14.966 | -4.85% |
| `ode_transformer_total` | 298.463 | 286.443 | -12.020 | -4.03% |
| `postprocessing` | 0.009 | 0.009 | +0.000 | +0.00% |
| `action_init_embed` | 2.024 | 0.343 | -1.681 | -83.06% |
| `action_init_noise` | 0.673 | 0.086 | -0.587 | -87.16% |
| `action_initialization` | 2.917 | 0.575 | -2.343 | -80.30% |
| `attention_mask_to_device` | 0.010 | 0.006 | -0.004 | -36.08% |
| `embed_tokens` | 0.354 | 0.058 | -0.297 | -83.75% |
| `image_cast` | 0.054 | 0.041 | -0.014 | -24.82% |
| `kv_cache_trim` | 0.868 | 0.847 | -0.021 | -2.39% |
| `moe_indices` | 0.151 | 0.101 | -0.050 | -33.23% |
| `ode_action_embed_total` | 3.380 | 2.938 | -0.441 | -13.06% |
| `ode_action_head_total` | 1.255 | 1.079 | -0.177 | -14.07% |
| `ode_prepare_inputs` | 0.758 | 0.686 | -0.072 | -9.45% |
| `position_ids_rope` | 0.835 | 0.000 | -0.835 | -100.00% |
| `postfix_mask_build` | 0.336 | 0.155 | -0.181 | -53.88% |
| `postfix_moe_indices` | 0.153 | 0.115 | -0.039 | -25.12% |
| `postfix_slice` | 0.060 | 0.057 | -0.003 | -5.39% |
| `prefill_action_head` | 0.268 | 0.177 | -0.091 | -33.92% |
| `prefix_length_resolve` | 4.995 | 0.102 | -4.893 | -97.97% |
| `pruning_position_ids_prepare` | 0.000 | 0.574 | +0.574 | +0.00% |
| `scatter_action_init` | 0.092 | 0.062 | -0.030 | -33.04% |
| `scatter_image_embeds` | 0.652 | 0.133 | -0.519 | -79.59% |
| `scatter_proprioception` | 2.303 | 0.161 | -2.142 | -93.01% |
| `vispruner_gather_image_embeds` | 0.000 | 0.042 | +0.042 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.062 | +0.062 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.080 | +0.080 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.135 | +0.135 | +0.00% |
| `vispruner_score_prepare` | 0.000 | 0.041 | +0.041 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 593.961 | 356.480 | -237.482 | -39.98% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/predictor_score_smoke_results.json`