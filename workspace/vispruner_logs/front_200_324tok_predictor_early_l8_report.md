# Wall-X VisPruner Front-Chain Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- samples: `200`
- warmup_samples: `5`
- keep_ratio: `0.5`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- image_min_pixels: `254016`
- image_max_pixels: `None`
- device: `cuda`

## Summary

- tokens: baseline `324.00`, pruned `162.00`
- front_direct_ms: baseline `129.231`, pruned `94.581`, delta `-34.650`
- front_top_level_ms: baseline `129.764`, pruned `95.023`, delta `-34.741`

## Stage Timings

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `external_prepare_batch_ms` | 7.955 | 7.546 | -0.409 | -5.14% |
| `embed_processing` | 97.200 | 64.069 | -33.131 | -34.09% |
| `image_path_total` | 96.692 | 63.682 | -33.010 | -34.14% |
| `vision_image_forward` | 96.720 | 63.712 | -33.008 | -34.13% |
| `vision_image_encode` | 96.586 | 0.000 | -96.586 | -100.00% |
| `vision_image_encode_score` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 62.234 | +62.234 | +0.00% |
| `vispruner_predictor_score` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_total` | 0.000 | 0.679 | +0.679 | +0.00% |
| `pruning_position_ids_prepare` | 0.000 | 0.598 | +0.598 | +0.00% |
| `scatter_image_embeds` | 0.157 | 0.100 | -0.057 | -36.08% |
| `position_encoding` | 0.839 | 0.114 | -0.725 | -86.41% |
| `prefetch_forward` | 31.724 | 30.840 | -0.885 | -2.79% |
| `prefill_transformer` | 31.543 | 30.685 | -0.858 | -2.72% |
| `prefill_action_head` | 0.130 | 0.107 | -0.023 | -18.04% |
| `total_time` | 130.350 | 95.543 | -34.807 | -26.70% |
