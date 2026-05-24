# Wall-X VisPruner 30 Image Timing Report

- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- image_dir: `/root/autodl-tmp/wall_x/benchmark_images/picsum_30`
- image_source: `https://picsum.photos/seed/wallx-vispruner-{idx}/640/480`
- num_images: `30`
- warmup: `1`
- iters: `3`
- keep_ratio: `0.5`
- device: `cuda`

## Summary

- Average vision tokens: baseline `63.00`, pruned `32.00`, reduction `49.21%`.
- Average model `total_time`: baseline `346.291 ms`, pruned `345.729 ms`, delta `-0.562 ms` (`-0.16%`).
- Note: timing was collected with `profile_timing=True`, so fine-grained CUDA synchronization overhead is included. Use paired baseline/pruned differences for diagnosis, not as online latency.

## Diagnostic Findings

- Image path did not become cheaper: `vision_image_forward` changed from `24.497 ms` to `24.742 ms`. In the pruned path, `vision_image_encode_score` alone costs `22.966 ms`, and `vispruner_total` adds `0.925 ms`.
- Prefix Transformer did not show a clear win in this run: `prefill_transformer` changed from `30.681 ms` to `30.931 ms`.
- The dominant ODE/postfix Transformer work is almost unchanged: `ode_transformer_total` changed from `280.841 ms` to `280.533 ms`. This explains why a 49% visual-token reduction does not translate into a large complete-action latency reduction.

## Average Timing By Segment

| segment | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 346.291 | 345.729 | -0.562 | -0.16% |
| `external_prepare_batch_ms` | 9.217 | 9.396 | +0.179 | +1.94% |
| `embed_processing` | 24.934 | 25.135 | +0.202 | +0.81% |
| `image_path_total` | 24.468 | 24.711 | +0.243 | +0.99% |
| `vision_image_forward` | 24.497 | 24.742 | +0.245 | +1.00% |
| `vision_image_encode` | 24.354 | 0.000 | -24.354 | -100.00% |
| `vision_image_encode_score` | 0.000 | 22.966 | +22.966 | +0.00% |
| `vispruner_total` | 0.000 | 0.925 | +0.925 | +0.00% |
| `vispruner_build_keep_mask` | 0.000 | 0.262 | +0.262 | +0.00% |
| `vispruner_topk_select` | 0.000 | 0.149 | +0.149 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.215 | +0.215 | +0.00% |
| `embed_tokens` | 0.046 | 0.045 | -0.001 | -1.57% |
| `scatter_image_embeds` | 0.139 | 0.111 | -0.028 | -20.14% |
| `position_encoding` | 0.698 | 0.120 | -0.578 | -82.82% |
| `position_ids_rope` | 0.552 | 0.000 | -0.552 | -100.00% |
| `moe_indices` | 0.095 | 0.096 | +0.001 | +0.74% |
| `action_initialization` | 0.472 | 0.474 | +0.001 | +0.30% |
| `prefetch_forward` | 30.905 | 31.142 | +0.237 | +0.77% |
| `prefill_transformer` | 30.681 | 30.931 | +0.250 | +0.82% |
| `prefill_action_head` | 0.160 | 0.150 | -0.010 | -6.35% |
| `cache_preprocessing` | 1.387 | 1.348 | -0.040 | -2.86% |
| `kv_cache_trim` | 0.875 | 0.873 | -0.002 | -0.19% |
| `postfix_mask_build` | 0.151 | 0.138 | -0.013 | -8.34% |
| `ode_integration` | 287.742 | 287.361 | -0.381 | -0.13% |
| `ode_action_embed_total` | 2.851 | 2.814 | -0.037 | -1.30% |
| `ode_prepare_inputs` | 0.671 | 0.664 | -0.007 | -1.00% |
| `ode_transformer_total` | 280.841 | 280.533 | -0.308 | -0.11% |
| `ode_action_head_total` | 1.031 | 1.015 | -0.017 | -1.61% |
| `postprocessing` | 0.010 | 0.009 | -0.000 | -2.90% |
| `action_init_embed` | 0.278 | 0.274 | -0.005 | -1.71% |
| `action_init_noise` | 0.052 | 0.050 | -0.001 | -2.86% |
| `attention_mask_to_device` | 0.007 | 0.007 | -0.000 | -2.78% |
| `image_cast` | 0.049 | 0.050 | +0.001 | +1.06% |
| `postfix_moe_indices` | 0.111 | 0.107 | -0.004 | -3.74% |
| `postfix_slice` | 0.057 | 0.058 | +0.001 | +1.73% |
| `prefix_length_resolve` | 0.082 | 0.076 | -0.006 | -7.51% |
| `pruning_position_ids_prepare` | 0.000 | 0.590 | +0.590 | +0.00% |
| `scatter_action_init` | 0.061 | 0.060 | -0.001 | -2.43% |
| `scatter_proprioception` | 0.141 | 0.135 | -0.006 | -4.36% |
| `vispruner_gather_image_embeds` | 0.000 | 0.043 | +0.043 | +0.00% |
| `vispruner_image_lengths` | 0.000 | 0.069 | +0.069 | +0.00% |
| `vispruner_pad_pruned_batch` | 0.000 | 0.080 | +0.080 | +0.00% |
| `vispruner_rope_deltas` | 0.000 | 0.137 | +0.137 | +0.00% |
| `vispruner_score_prepare` | 0.000 | 0.032 | +0.032 | +0.00% |

## Per Image Paired Results

| idx | image | token_before | token_after | token_reduction | baseline_total_ms | pruned_total_ms | delta_ms | delta_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `picsum_000.jpg` | 63 | 32 | 49.21% | 347.490 | 349.419 | +1.929 | +0.56% |
| 2 | `picsum_001.jpg` | 63 | 32 | 49.21% | 346.515 | 347.623 | +1.107 | +0.32% |
| 3 | `picsum_002.jpg` | 63 | 32 | 49.21% | 344.589 | 344.727 | +0.138 | +0.04% |
| 4 | `picsum_003.jpg` | 63 | 32 | 49.21% | 345.831 | 347.272 | +1.440 | +0.42% |
| 5 | `picsum_004.jpg` | 63 | 32 | 49.21% | 354.449 | 346.357 | -8.091 | -2.28% |
| 6 | `picsum_005.jpg` | 63 | 32 | 49.21% | 354.682 | 343.622 | -11.060 | -3.12% |
| 7 | `picsum_006.jpg` | 63 | 32 | 49.21% | 350.350 | 344.518 | -5.832 | -1.66% |
| 8 | `picsum_007.jpg` | 63 | 32 | 49.21% | 348.255 | 341.965 | -6.291 | -1.81% |
| 9 | `picsum_008.jpg` | 63 | 32 | 49.21% | 344.535 | 340.741 | -3.794 | -1.10% |
| 10 | `picsum_009.jpg` | 63 | 32 | 49.21% | 342.402 | 349.502 | +7.100 | +2.07% |
| 11 | `picsum_010.jpg` | 63 | 32 | 49.21% | 343.878 | 348.752 | +4.874 | +1.42% |
| 12 | `picsum_011.jpg` | 63 | 32 | 49.21% | 342.213 | 343.465 | +1.252 | +0.37% |
| 13 | `picsum_012.jpg` | 63 | 32 | 49.21% | 342.996 | 343.565 | +0.569 | +0.17% |
| 14 | `picsum_013.jpg` | 63 | 32 | 49.21% | 341.849 | 342.078 | +0.229 | +0.07% |
| 15 | `picsum_014.jpg` | 63 | 32 | 49.21% | 344.384 | 342.358 | -2.026 | -0.59% |
| 16 | `picsum_015.jpg` | 63 | 32 | 49.21% | 347.788 | 341.979 | -5.810 | -1.67% |
| 17 | `picsum_016.jpg` | 63 | 32 | 49.21% | 344.178 | 346.692 | +2.515 | +0.73% |
| 18 | `picsum_017.jpg` | 63 | 32 | 49.21% | 347.007 | 347.325 | +0.319 | +0.09% |
| 19 | `picsum_018.jpg` | 63 | 32 | 49.21% | 347.331 | 343.983 | -3.348 | -0.96% |
| 20 | `picsum_019.jpg` | 63 | 32 | 49.21% | 344.482 | 346.352 | +1.869 | +0.54% |
| 21 | `picsum_020.jpg` | 63 | 32 | 49.21% | 343.889 | 350.059 | +6.170 | +1.79% |
| 22 | `picsum_021.jpg` | 63 | 32 | 49.21% | 345.111 | 350.387 | +5.276 | +1.53% |
| 23 | `picsum_022.jpg` | 63 | 32 | 49.21% | 344.359 | 349.028 | +4.669 | +1.36% |
| 24 | `picsum_023.jpg` | 63 | 32 | 49.21% | 342.195 | 345.781 | +3.587 | +1.05% |
| 25 | `picsum_024.jpg` | 63 | 32 | 49.21% | 350.557 | 347.355 | -3.201 | -0.91% |
| 26 | `picsum_025.jpg` | 63 | 32 | 49.21% | 354.698 | 346.584 | -8.114 | -2.29% |
| 27 | `picsum_026.jpg` | 63 | 32 | 49.21% | 348.733 | 346.660 | -2.073 | -0.59% |
| 28 | `picsum_027.jpg` | 63 | 32 | 49.21% | 344.995 | 346.367 | +1.372 | +0.40% |
| 29 | `picsum_028.jpg` | 63 | 32 | 49.21% | 344.500 | 343.319 | -1.180 | -0.34% |
| 30 | `picsum_029.jpg` | 63 | 32 | 49.21% | 344.484 | 344.035 | -0.449 | -0.13% |

## Timing Counts

These counts show how many times repeated timing blocks were accumulated per measured run.

| case | segment | count |
|---|---|---:|
| `baseline` | `action_init_embed` | 1 |
| `baseline` | `action_init_noise` | 1 |
| `baseline` | `action_initialization` | 1 |
| `baseline` | `attention_mask_to_device` | 1 |
| `baseline` | `cache_preprocessing` | 1 |
| `baseline` | `embed_processing` | 1 |
| `baseline` | `embed_tokens` | 1 |
| `baseline` | `image_cast` | 1 |
| `baseline` | `image_path_total` | 1 |
| `baseline` | `kv_cache_trim` | 1 |
| `baseline` | `moe_indices` | 1 |
| `baseline` | `ode_action_embed_total` | 9 |
| `baseline` | `ode_action_head_total` | 9 |
| `baseline` | `ode_integration` | 1 |
| `baseline` | `ode_prepare_inputs` | 9 |
| `baseline` | `ode_transformer_total` | 9 |
| `baseline` | `position_encoding` | 1 |
| `baseline` | `position_ids_rope` | 1 |
| `baseline` | `postfix_mask_build` | 1 |
| `baseline` | `postfix_moe_indices` | 1 |
| `baseline` | `postfix_slice` | 1 |
| `baseline` | `postprocessing` | 1 |
| `baseline` | `prefetch_forward` | 1 |
| `baseline` | `prefill_action_head` | 1 |
| `baseline` | `prefill_transformer` | 1 |
| `baseline` | `prefix_length_resolve` | 1 |
| `baseline` | `scatter_action_init` | 1 |
| `baseline` | `scatter_image_embeds` | 1 |
| `baseline` | `scatter_proprioception` | 1 |
| `baseline` | `total_time` | 1 |
| `baseline` | `vision_image_encode` | 1 |
| `baseline` | `vision_image_forward` | 1 |
| `pruned` | `action_init_embed` | 1 |
| `pruned` | `action_init_noise` | 1 |
| `pruned` | `action_initialization` | 1 |
| `pruned` | `attention_mask_to_device` | 1 |
| `pruned` | `cache_preprocessing` | 1 |
| `pruned` | `embed_processing` | 1 |
| `pruned` | `embed_tokens` | 1 |
| `pruned` | `image_cast` | 1 |
| `pruned` | `image_path_total` | 1 |
| `pruned` | `kv_cache_trim` | 1 |
| `pruned` | `moe_indices` | 1 |
| `pruned` | `ode_action_embed_total` | 9 |
| `pruned` | `ode_action_head_total` | 9 |
| `pruned` | `ode_integration` | 1 |
| `pruned` | `ode_prepare_inputs` | 9 |
| `pruned` | `ode_transformer_total` | 9 |
| `pruned` | `position_encoding` | 1 |
| `pruned` | `postfix_mask_build` | 1 |
| `pruned` | `postfix_moe_indices` | 1 |
| `pruned` | `postfix_slice` | 1 |
| `pruned` | `postprocessing` | 1 |
| `pruned` | `prefetch_forward` | 1 |
| `pruned` | `prefill_action_head` | 1 |
| `pruned` | `prefill_transformer` | 1 |
| `pruned` | `prefix_length_resolve` | 1 |
| `pruned` | `pruning_position_ids_prepare` | 1 |
| `pruned` | `scatter_action_init` | 1 |
| `pruned` | `scatter_image_embeds` | 1 |
| `pruned` | `scatter_proprioception` | 1 |
| `pruned` | `total_time` | 1 |
| `pruned` | `vision_image_encode_score` | 1 |
| `pruned` | `vision_image_forward` | 1 |
| `pruned` | `vispruner_apply_keep_to_sequences` | 1 |
| `pruned` | `vispruner_build_keep_mask` | 1 |
| `pruned` | `vispruner_gather_image_embeds` | 1 |
| `pruned` | `vispruner_image_lengths` | 1 |
| `pruned` | `vispruner_pad_pruned_batch` | 1 |
| `pruned` | `vispruner_rope_deltas` | 1 |
| `pruned` | `vispruner_score_prepare` | 1 |
| `pruned` | `vispruner_topk_select` | 1 |
| `pruned` | `vispruner_total` | 1 |

## Raw Results JSON

- `/root/autodl-tmp/wall_x/wall_x/report/wallx_vispruner_30image_timing_results.json`
