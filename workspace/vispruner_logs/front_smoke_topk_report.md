# Wall-X VisPruner Front-Chain Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- samples: `2`
- warmup_samples: `1`
- keep_ratio: `0.5`
- pruned_strategy: `topk_attention`
- predictor_checkpoint: `None`
- predictor_source: `early_hidden`
- predictor_early_layer: `None`
- image_min_pixels: `None`
- image_max_pixels: `None`
- device: `cuda`

## Summary

- tokens: baseline `81.00`, pruned `41.00`
- front_direct_ms: baseline `116.181`, pruned `115.280`, delta `-0.901`
- front_top_level_ms: baseline `119.151`, pruned `117.255`, delta `-1.896`

## Stage Timings

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `external_prepare_batch_ms` | 5.670 | 5.182 | -0.488 | -8.61% |
| `embed_processing` | 84.050 | 85.484 | +1.434 | +1.71% |
| `image_path_total` | 82.072 | 84.548 | +2.476 | +3.02% |
| `vision_image_forward` | 82.255 | 84.669 | +2.414 | +2.94% |
| `vision_image_encode` | 81.115 | 0.000 | -81.115 | -100.00% |
| `vision_image_encode_score` | 0.000 | 79.724 | +79.724 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_predictor_score` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_total` | 0.000 | 2.808 | +2.808 | +0.00% |
| `pruning_position_ids_prepare` | 0.000 | 1.278 | +1.278 | +0.00% |
| `scatter_image_embeds` | 0.190 | 0.147 | -0.043 | -22.39% |
| `position_encoding` | 3.190 | 0.357 | -2.834 | -88.82% |
| `prefetch_forward` | 31.911 | 31.414 | -0.497 | -1.56% |
| `prefill_transformer` | 30.729 | 30.227 | -0.502 | -1.63% |
| `prefill_action_head` | 0.243 | 0.181 | -0.062 | -25.61% |
| `total_time` | 121.156 | 118.524 | -2.632 | -2.17% |
