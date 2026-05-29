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
- image_min_pixels: `None`
- image_max_pixels: `None`
- device: `cuda`

## Summary

- tokens: baseline `81.00`, pruned `41.00`
- front_direct_ms: baseline `57.209`, pruned `57.487`, delta `+0.278`
- front_top_level_ms: baseline `57.648`, pruned `57.921`, delta `+0.272`

## Stage Timings

| stage | baseline_ms | pruned_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `external_prepare_batch_ms` | 3.725 | 3.749 | +0.024 | +0.64% |
| `embed_processing` | 27.427 | 28.942 | +1.515 | +5.52% |
| `image_path_total` | 27.033 | 28.569 | +1.536 | +5.68% |
| `vision_image_forward` | 27.059 | 28.597 | +1.538 | +5.68% |
| `vision_image_encode` | 26.943 | 0.000 | -26.943 | -100.00% |
| `vision_image_encode_score` | 0.000 | 27.045 | +27.045 | +0.00% |
| `vision_image_encode_early_prune` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_predictor_score` | 0.000 | 0.000 | +0.000 | +0.00% |
| `vispruner_total` | 0.000 | 0.830 | +0.830 | +0.00% |
| `pruning_position_ids_prepare` | 0.000 | 0.528 | +0.528 | +0.00% |
| `scatter_image_embeds` | 0.112 | 0.097 | -0.016 | -13.96% |
| `position_encoding` | 0.620 | 0.124 | -0.496 | -80.01% |
| `prefetch_forward` | 29.602 | 28.855 | -0.747 | -2.52% |
| `prefill_transformer` | 29.443 | 28.698 | -0.746 | -2.53% |
| `prefill_action_head` | 0.105 | 0.103 | -0.002 | -1.45% |
| `total_time` | 58.148 | 58.424 | +0.276 | +0.47% |
