# Wall-X Original vs Predictor Action Output Comparison

- samples: `300`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source: `early_hidden`
- predictor_early_layer: `8`
- seed: `20260605`
- allclose atol/rtol: `0.001` / `0.001`

## Summary

- tokens: baseline `324.00`, pruned `162.00`
- total_time_ms: baseline `0.000`, pruned `0.000`, delta `+0.000` (`+0.00%`)
- action MAE: `0.025359`
- action RMSE: `0.031705`
- action mean max_abs: `0.100451`
- action worst max_abs: `0.203187`
- action cosine_similarity: `0.999958`
- action allclose_rate: `0.0000`

## Metric Meaning

| metric | meaning |
|---|---|
| `MAE` | Mean absolute difference between predictor and original action tensors. Lower is closer. |
| `RMSE` | Root mean squared difference between action tensors. Lower is closer. |
| `max_abs` | Maximum absolute element-wise action difference per sample. Lower is closer. |
| `cosine_similarity` | Direction similarity between flattened action tensors. Closer to 1 is better. |
| `allclose_rate` | Fraction of samples passing `torch.allclose` under the configured atol/rtol. |
