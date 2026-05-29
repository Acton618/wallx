# Wall-X VisPruner Front-Chain Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- samples: `200`
- warmup_samples: `5`
- keep_ratio: `0.5`
- pruned_strategy: `topk_attention`
- predictor_checkpoint: `None`
- predictor_source: `early_hidden`
- predictor_early_layer: `None`
- image_min_pixels: `254016`
- image_max_pixels: `None`
- device: `cuda`

## Summary

- tokens: baseline `324.00`, pruned `162.00`
- front_direct_ms: baseline `128.312`, pruned `130.496`, delta `+2.184`
- front_top_level_ms: baseline `128.771`, pruned `130.957`, delta `+2.186`

## Stage Timings

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `external_prepare_batch_ms` | 7.601 | 8.679 | +1.078 | +14.18% |
| `embed_processing` | 96.960 | 99.321 | +2.361 | +2.44% |
| `image_path_total` | 96.535 | 98.390 | +1.855 | +1.92% |
| `vision_image_forward` | 96.561 | 98.418 | +1.857 | +1.92% |
| `vision_image_encode` | 96.441 | 0.000 | -96.441 | -100.00% |
| `vision_image_encode_score` | 0.000 | 96.619 | +96.619 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_predictor_score` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_total` | 0.000 | 0.907 | +0.907 | +0.00% |
| `pruning_position_ids_prepare` | 0.000 | 0.680 | +0.680 | +0.00% |
| `scatter_image_embeds` | 0.123 | 0.624 | +0.502 | +408.02% |
| `position_encoding` | 0.695 | 0.116 | -0.579 | -83.30% |
| `prefetch_forward` | 31.116 | 31.520 | +0.404 | +1.30% |
| `prefill_transformer` | 30.959 | 31.365 | +0.406 | +1.31% |
| `prefill_action_head` | 0.107 | 0.107 | -0.000 | -0.26% |
| `total_time` | 129.303 | 131.498 | +2.195 | +1.70% |
