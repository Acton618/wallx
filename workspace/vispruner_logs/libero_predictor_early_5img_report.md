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
- device: `cuda`

> Note: current VisPruner hard-pruning is wired to image tokens only. Video samples are processed through the model video path and timed, but video tokens are not pruned by the current implementation.

## Image Samples

- samples: `5`
- tokens: baseline `81.00`, pruned `41.00`, delta `-40.00` (`-49.38%`)
- total_time: baseline `360.288 ms`, pruned `345.280 ms`, delta `-15.008 ms` (`-4.17%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 360.288 | 345.280 | -15.008 | -4.17% |
| `external_prepare_batch_ms` | 4.059 | 3.443 | -0.616 | -15.17% |
| `embed_processing` | 27.960 | 30.799 | +2.839 | +10.15% |
| `image_path_total` | 27.470 | 30.377 | +2.906 | +10.58% |
| `vision_image_forward` | 27.498 | 30.407 | +2.909 | +10.58% |
| `vision_image_encode` | 27.353 | 0.000 | -27.353 | -100.00% |
| `vispruner_total` | 0.000 | 0.802 | +0.802 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.060 | +0.060 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 28.806 | +28.806 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.257 | +0.257 | +0.00% |
| `position_encoding` | 0.735 | 0.116 | -0.620 | -84.27% |
| `prefetch_forward` | 30.751 | 29.916 | -0.835 | -2.71% |
| `prefill_transformer` | 30.485 | 29.709 | -0.775 | -2.54% |
| `cache_preprocessing` | 1.422 | 1.296 | -0.125 | -8.80% |
| `ode_integration` | 298.768 | 282.567 | -16.200 | -5.42% |
| `ode_transformer_total` | 290.537 | 275.895 | -14.643 | -5.04% |
| `postprocessing` | 0.010 | 0.009 | -0.000 | -4.85% |
| `action_init_embed` | 0.310 | 0.268 | -0.042 | -13.50% |
| `action_init_noise` | 0.055 | 0.050 | -0.005 | -9.56% |
| `action_initialization` | 0.506 | 0.449 | -0.057 | -11.30% |
| `attention_mask_to_device` | 0.007 | 0.007 | -0.000 | -3.92% |
| `embed_tokens` | 0.055 | 0.050 | -0.005 | -9.81% |
| `image_cast` | 0.055 | 0.049 | -0.007 | -11.93% |
| `kv_cache_trim` | 0.889 | 0.836 | -0.053 | -6.00% |
| `moe_indices` | 0.099 | 0.095 | -0.004 | -4.22% |
| `ode_action_embed_total` | 3.385 | 2.767 | -0.619 | -18.27% |
| `ode_action_head_total` | 1.401 | 1.011 | -0.390 | -27.82% |
| `ode_prepare_inputs` | 0.760 | 0.654 | -0.106 | -13.93% |
| `position_ids_rope` | 0.589 | 0.000 | -0.589 | -100.00% |
| `postfix_mask_build` | 0.155 | 0.134 | -0.021 | -13.28% |
| `postfix_moe_indices` | 0.131 | 0.104 | -0.026 | -20.13% |
| `postfix_slice` | 0.059 | 0.056 | -0.003 | -5.15% |
| `prefill_action_head` | 0.198 | 0.150 | -0.048 | -24.03% |
| `prefix_length_resolve` | 0.093 | 0.081 | -0.011 | -12.42% |
| `pruning_position_ids_prepare` | 0.000 | 0.581 | +0.581 | +0.00% |
| `scatter_action_init` | 0.063 | 0.057 | -0.006 | -9.81% |
| `scatter_image_embeds` | 0.152 | 0.115 | -0.037 | -24.11% |
| `scatter_proprioception` | 0.155 | 0.132 | -0.023 | -14.55% |
| `vispruner_gather_image_embeds` | 0.000 | 0.030 | +0.030 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.084 | +0.084 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.101 | +0.101 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.142 | +0.142 | +0.00% |

## Video Samples

- samples: `0`
- tokens: baseline `0.00`, pruned `0.00`, delta `+0.00` (`+0.00%`)
- total_time: baseline `0.000 ms`, pruned `0.000 ms`, delta `+0.000 ms` (`+0.00%`)

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|

## Image Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 347.254 | 349.698 | +2.443 | +0.70% |
| 2 | `dataset_index=67942 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 351.267 | 343.132 | -8.135 | -2.32% |
| 3 | `dataset_index=135885 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 352.838 | 343.788 | -9.051 | -2.57% |
| 4 | `dataset_index=203828 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 355.536 | 345.847 | -9.689 | -2.73% |
| 5 | `dataset_index=271771 image_key=observation.images.faceImg` | 81 | 41 | -49.38% | 394.544 | 343.937 | -50.607 | -12.83% |

## Video Paired Samples

| idx | source | tokens_before | tokens_after | token_delta_pct | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|

## Raw Results

- `workspace/vispruner_logs/libero_predictor_early_5img_results.json`