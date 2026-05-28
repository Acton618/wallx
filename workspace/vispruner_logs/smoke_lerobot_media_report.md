# Wall-X LeRobot Media VisPruner Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `1`
- num_videos: `1`
- video_frames_per_sample: `4`
- warmup: `0`
- iters: `1`
- keep_ratio: `0.5`
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `1`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `871.390 ms`, pruned `635.650 ms`, delta `-235.740 ms` (`-27.05%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 871.390 | 635.650 | -235.740 | -27.05% |
| `external_prepare_batch_ms` | 16.597 | 11.234 | -5.363 | -32.31% |
| `embed_processing` | 377.224 | 118.077 | -259.148 | -68.70% |
| `image_path_total` | 352.325 | 106.364 | -245.960 | -69.81% |
| `vision_image_forward` | 354.595 | 106.400 | -248.196 | -69.99% |
| `vision_image_encode` | 345.916 | 0.000 | -345.916 | -100.00% |
| `vision_image_encode_score` | 0.000 | 104.676 | +104.676 | +0.00% |
| `vispruner_total` | 0.000 | 0.895 | +0.895 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.256 | +0.256 | +0.00% |
| `vispruner_topk_select` | 0.000 | 0.143 | +0.143 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.206 | +0.206 | +0.00% |
| `position_encoding` | 24.298 | 4.164 | -20.134 | -82.86% |
| `prefetch_forward` | 54.258 | 36.722 | -17.536 | -32.32% |
| `prefill_transformer` | 42.922 | 31.794 | -11.128 | -25.93% |
| `cache_preprocessing` | 3.431 | 18.107 | +14.676 | +427.69% |
| `ode_integration` | 387.453 | 445.868 | +58.415 | +15.08% |
| `ode_transformer_total` | 279.418 | 281.507 | +2.088 | +0.75% |
| `postprocessing` | 0.009 | 0.001 | -0.008 | -88.89% |
| `action_init_embed` | 6.187 | 0.404 | -5.783 | -93.46% |
| `action_init_noise` | 5.835 | 0.103 | -5.731 | -98.23% |
| `action_initialization` | 18.918 | 1.237 | -17.681 | -93.46% |
| `attention_mask_to_device` | 0.002 | 0.002 | +0.000 | +0.00% |
| `embed_tokens` | 0.009 | 0.189 | +0.180 | +1952.43% |
| `image_cast` | 1.899 | 0.046 | -1.853 | -97.58% |
| `kv_cache_trim` | 0.851 | 0.884 | +0.032 | +3.81% |
| `moe_indices` | 2.334 | 1.869 | -0.465 | -19.92% |
| `ode_action_embed_total` | 10.107 | 12.307 | +2.201 | +21.78% |
| `ode_action_head_total` | 13.028 | 18.485 | +5.457 | +41.88% |
| `ode_prepare_inputs` | 9.647 | 11.916 | +2.268 | +23.51% |
| `position_ids_rope` | 17.421 | 0.000 | -17.421 | -100.00% |
| `postfix_mask_build` | 0.341 | 6.731 | +6.390 | +1872.22% |
| `postfix_moe_indices` | 0.140 | 3.594 | +3.454 | +2463.57% |
| `postfix_slice` | 0.058 | 0.059 | +0.001 | +2.06% |
| `prefill_action_head` | 4.553 | 2.413 | -2.140 | -47.01% |
| `prefix_length_resolve` | 1.935 | 0.014 | -1.921 | -99.26% |
| `pruning_position_ids_prepare` | 0.000 | 0.597 | +0.597 | +0.00% |
| `scatter_action_init` | 2.280 | 0.170 | -2.110 | -92.55% |
| `scatter_image_embeds` | 4.554 | 2.424 | -2.129 | -46.76% |
| `scatter_proprioception` | 6.813 | 2.278 | -4.534 | -66.56% |
| `vispruner_gather_image_embeds` | 0.000 | 0.043 | +0.043 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.065 | +0.065 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.080 | +0.080 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.137 | +0.137 | +0.00% |
| `vispruner_score_prepare` | 0.000 | 0.034 | +0.034 | +0.00% |

## Video Samples

- samples: `1`
- tokens: baseline `162.00`, pruned `162.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `1044.535 ms`, pruned `376.366 ms`, delta `-668.169 ms` (`-63.97%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 1044.535 | 376.366 | -668.169 | -63.97% |
| `external_prepare_batch_ms` | 32.893 | 16.096 | -16.797 | -51.06% |
| `embed_processing` | 569.515 | 39.619 | -529.896 | -93.04% |
| `vision_video_forward` | 562.458 | 38.953 | -523.505 | -93.07% |
| `scatter_video_embeds` | 0.094 | 0.085 | -0.009 | -9.78% |
| `position_encoding` | 0.821 | 0.771 | -0.050 | -6.09% |
| `prefetch_forward` | 42.509 | 34.214 | -8.296 | -19.51% |
| `prefill_transformer` | 41.869 | 34.002 | -7.867 | -18.79% |
| `cache_preprocessing` | 2.094 | 1.372 | -0.722 | -34.47% |
| `ode_integration` | 428.537 | 299.625 | -128.912 | -30.08% |
| `ode_transformer_total` | 319.124 | 292.793 | -26.332 | -8.25% |
| `postprocessing` | 0.009 | 0.009 | -0.000 | -2.38% |
| `action_init_embed` | 0.385 | 0.365 | -0.020 | -5.13% |
| `action_init_noise` | 0.095 | 0.091 | -0.005 | -4.81% |
| `action_initialization` | 0.637 | 0.607 | -0.030 | -4.71% |
| `attention_mask_to_device` | 0.007 | 0.007 | -0.000 | -1.32% |
| `embed_tokens` | 2.039 | 0.100 | -1.939 | -95.08% |
| `kv_cache_trim` | 0.832 | 0.847 | +0.015 | +1.80% |
| `moe_indices` | 0.106 | 0.105 | -0.001 | -1.12% |
| `ode_action_embed_total` | 10.554 | 2.788 | -7.766 | -73.58% |
| `ode_action_head_total` | 9.823 | 1.025 | -8.798 | -89.57% |
| `ode_prepare_inputs` | 5.536 | 0.692 | -4.843 | -87.49% |
| `position_ids_rope` | 0.668 | 0.621 | -0.047 | -7.05% |
| `postfix_mask_build` | 0.398 | 0.156 | -0.242 | -60.83% |
| `postfix_moe_indices` | 0.284 | 0.123 | -0.161 | -56.56% |
| `postfix_slice` | 0.058 | 0.056 | -0.002 | -4.17% |
| `prefill_action_head` | 0.328 | 0.161 | -0.167 | -50.94% |
| `prefix_length_resolve` | 0.013 | 0.102 | +0.088 | +662.74% |
| `scatter_action_init` | 0.066 | 0.065 | -0.001 | -1.21% |
| `scatter_proprioception` | 0.209 | 0.216 | +0.007 | +3.43% |

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 871.390 | 635.650 | -235.740 | -27.05% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `episode_000000.mp4` | 162 | 162 | +0.00% | 1044.535 | 376.366 | -668.169 | -63.97% |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/smoke_lerobot_media_results.json`