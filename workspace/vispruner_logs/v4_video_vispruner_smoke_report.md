# Wall-X V4 Video VisPruner + V3 ODE Early Stop Video Dataset Report

- video_dir: `/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg`
- video_glob: `episode_*.mp4`
- num_videos: `1`
- video_frames_per_clip: `4`
- prompt: `pick up the object`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- media_type: `video`
- vispruner.prune_video: `True`
- vispruner.keep_ratio: `0.5`
- num_inference_timesteps: `10`
- warmup: `0`
- iters: `1`
- base_seed: `1234`
- device: `cuda`

## V3 Cases

| case | enable | threshold | min_steps | patience | metric |
|---|---:|---:|---:|---:|---|
| `fixed_10` | `False` | `-` | `-` | `-` | `mean_abs` |
| `early_safe` | `True` | `0.2` | `2` | `1` | `mean_abs` |
| `early_tradeoff` | `True` | `0.3` | `8` | `1` | `mean_abs` |

## Summary

| case | video_tokens_before | expected_video_tokens_after | total_ms | total_delta_vs_fixed | ode_ms | ode_delta_vs_fixed | actual_updates | postfix_steps | stopped_rate | action_mae_vs_fixed | action_rmse_vs_fixed | action_max_abs_vs_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 162.00 | 81.00 | 616.485 | +0.00% | 291.017 | +0.00% | 10.00 | 9.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `early_safe` | 162.00 | 81.00 | 374.287 | -39.29% | 296.525 | +1.89% | 10.00 | 9.00 | 0.00% | 0.000000 | 0.000000 | 0.000000 |
| `early_tradeoff` | 162.00 | 81.00 | 297.608 | -51.73% | 221.500 | -23.89% | 8.00 | 7.00 | 100.00% | 0.520685 | 0.665823 | 1.480715 |

## Interpretation

- This report uses MP4 video clips only. The model input contains `pixel_values_videos`, `video_grid_thw`, and explicit `second_per_grid_ts` from decoded FPS.
- `expected_video_tokens_after` is the V4 internal video token count after VisPruner. The raw batch still contains the original placeholders before the model prunes them.
- Accuracy is action difference against `fixed_10` under the same video sample and seed; it measures how much V3 early stop changes the original fixed-step inference output.
- `actual_updates` counts the existing prefetch update plus later postfix ODE updates. `postfix_steps` is `actual_updates - 1`.
- Fine-grained timings use `profile_timing=True`, so use paired deltas rather than absolute latency as the main signal.

## Stage Timing

| stage | fixed_10_ms | early_safe_ms | early_tradeoff_ms | tradeoff_delta_vs_fixed |
|---|---:|---:|---:|---:|
| `total_time` | 616.485 | 374.287 | 297.608 | -51.73% |
| `external_prepare_batch_ms` | 19.429 | 15.093 | 15.043 | -22.57% |
| `embed_processing` | 282.450 | 45.435 | 44.940 | -84.09% |
| `vision_video_forward` | 279.878 | 45.034 | 44.556 | -84.08% |
| `scatter_video_embeds` | 0.648 | 0.116 | 0.109 | -83.14% |
| `position_encoding` | 0.203 | 0.122 | 0.122 | -39.81% |
| `action_initialization` | 3.008 | 0.469 | 0.448 | -85.09% |
| `prefetch_forward` | 36.132 | 30.258 | 29.113 | -19.42% |
| `prefill_transformer` | 35.806 | 30.044 | 28.910 | -19.26% |
| `cache_preprocessing` | 3.517 | 1.334 | 1.326 | -62.29% |
| `ode_integration` | 291.017 | 296.525 | 221.500 | -23.89% |
| `ode_transformer_total` | 282.937 | 290.375 | 216.865 | -23.35% |
| `ode_action_embed_total` | 2.916 | 2.789 | 2.125 | -27.12% |
| `ode_prepare_inputs` | 0.693 | 0.687 | 0.525 | -24.22% |
| `ode_action_head_total` | 1.020 | 1.028 | 0.777 | -23.81% |
| `postprocessing` | 0.007 | 0.007 | 0.007 | +1.42% |
| `action_init_embed` | 2.058 | 0.270 | 0.263 | -87.21% |
| `action_init_noise` | 0.709 | 0.052 | 0.051 | -92.82% |
| `attention_mask_to_device` | 0.008 | 0.006 | 0.007 | -12.50% |
| `embed_tokens` | 0.346 | 0.053 | 0.046 | -86.74% |
| `kv_cache_trim` | 0.877 | 0.858 | 0.852 | -2.90% |
| `moe_indices` | 0.179 | 0.100 | 0.099 | -45.08% |
| `postfix_mask_build` | 0.401 | 0.136 | 0.142 | -64.60% |
| `postfix_moe_indices` | 0.144 | 0.113 | 0.104 | -28.29% |
| `postfix_slice` | 0.058 | 0.056 | 0.062 | +7.88% |
| `prefill_action_head` | 0.263 | 0.157 | 0.147 | -43.98% |
| `prefix_length_resolve` | 1.928 | 0.081 | 0.076 | -96.07% |
| `scatter_action_init` | 0.089 | 0.068 | 0.058 | -34.44% |
| `scatter_proprioception` | 1.451 | 0.134 | 0.127 | -91.25% |
| `video_cast` | 0.089 | 0.057 | 0.074 | -16.93% |
| `video_path_total` | 279.842 | 45.005 | 44.520 | -84.09% |
| `video_pruning_position_ids_prepare` | 4.336 | 0.674 | 0.612 | -85.88% |
| `vision_video_encode_score` | 261.336 | 43.152 | 42.755 | -83.64% |
| `vispruner_apply_keep_to_sequences` | 0.304 | 0.246 | 0.205 | -32.66% |
| `vispruner_build_keep_mask` | 9.908 | 0.277 | 0.267 | -97.31% |
| `vispruner_gather_image_embeds` | 0.075 | 0.047 | 0.044 | -41.40% |
| `vispruner_image_lengths` | 3.052 | 0.083 | 0.084 | -97.24% |
| `vispruner_pad_pruned_batch` | 0.126 | 0.083 | 0.083 | -34.49% |
| `vispruner_rope_deltas` | 0.262 | 0.137 | 0.129 | -50.89% |
| `vispruner_score_prepare` | 6.042 | 0.057 | 0.051 | -99.16% |
| `vispruner_topk_select` | 3.739 | 0.148 | 0.139 | -96.28% |
| `vispruner_total` | 13.867 | 0.982 | 0.921 | -93.36% |

## Per-Video Paired Results

| idx | source | video_grid_thw | second_per_grid_ts | safe_updates | tradeoff_updates | fixed_total_ms | safe_total_ms | tradeoff_total_ms | safe_mae | tradeoff_mae |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `video=/root/autodl-tmp/wall_x/datasheet/libero_all/videos/chunk-000/observation.images.faceImg/episode_000000.mp4` | `[[2, 18, 18]]` | `[0.06666667014360428]` | 10.00 | 8.00 | 616.485 | 374.287 | 297.608 | 0.000000 | 0.520685 |

## Raw Results
- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/v4_video_vispruner_smoke_results.json`