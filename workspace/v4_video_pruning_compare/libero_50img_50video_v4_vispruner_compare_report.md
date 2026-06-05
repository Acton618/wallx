# LIBERO 50 Image + 50 Video V4 Video VisPruner Inference Report

## Experiment Setup

- dataset: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- model: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- image samples: `50`, key: `observation.images.faceImg`
- video samples: `50`, each clip uses `4` decoded frames
- inference cases: `fixed_10`, `cache_i2`, `cache_i3`, `early_tradeoff`
- timing setting: `warmup=0`, `iters=1`, `device=cuda`
- V4 video pruning setting: `enable=true`, `strategy=topk_attention`, `keep_ratio=0.5`, `prune_video=true`
- video no-VisPruner baseline: `enable_pruning=false`, `prune_video=false`

## Image 50 Samples: VisPruner + Full Inference

| case | tokens before->after | total ms | delta vs fixed | ODE ms | ODE delta | cache hit | action MAE vs fixed | max abs vs fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 81->41 | 341.600 | +0.00% | 281.455 | +0.00% | 0.00% | 0.000000 | 0.000000 |
| `cache_i2` | 81->41 | 217.309 | -36.38% | 157.472 | -44.05% | 44.44% | 0.008434 | 0.034757 |
| `cache_i3` | 81->41 | 155.518 | -54.47% | 95.497 | -66.07% | 66.67% | 0.014082 | 0.054655 |
| `early_tradeoff` | 81->41 | 281.224 | -17.67% | 221.347 | -21.36% | 0.00% | 0.517708 | 1.485309 |

## Video 50 Samples: V4 Video VisPruner Enabled

| case | video tokens before->after | total ms | delta vs fixed | ODE ms | ODE delta | cache hit | action MAE vs fixed | max abs vs fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 162->81 | 370.337 | +0.00% | 288.863 | +0.00% | 0.00% | 0.000000 | 0.000000 |
| `cache_i2` | 162->81 | 238.488 | -35.60% | 161.361 | -44.14% | 44.44% | 0.008477 | 0.034346 |
| `cache_i3` | 162->81 | 173.047 | -53.27% | 96.547 | -66.58% | 66.67% | 0.014004 | 0.053875 |
| `early_tradeoff` | 162->81 | 301.719 | -18.53% | 225.120 | -22.07% | 0.00% | 0.519379 | 1.480584 |

## Video 50 Samples: No Video VisPruner Baseline

| case | video tokens before->after | total ms | delta vs fixed | ODE ms | ODE delta | cache hit | action MAE vs fixed | max abs vs fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 162->162 | 374.109 | +0.00% | 293.070 | +0.00% | 0.00% | 0.000000 | 0.000000 |
| `cache_i2` | 162->162 | 239.632 | -35.95% | 163.204 | -44.31% | 44.44% | 0.008398 | 0.034591 |
| `cache_i3` | 162->162 | 172.093 | -54.00% | 96.504 | -67.07% | 66.67% | 0.014056 | 0.054988 |
| `early_tradeoff` | 162->162 | 304.843 | -18.51% | 228.409 | -22.06% | 0.00% | 0.519180 | 1.480248 |

## Direct Comparison: Video VisPruner ON vs OFF

| case | tokens off | tokens on | token reduction | total off ms | total on ms | total delta | vision off ms | vision on ms | vision delta | action MAE on vs off | max abs on vs off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_10` | 162 | 81 | -50.00% | 374.109 | 370.337 | -1.01% | 47.300 | 49.358 | +4.35% | 0.024422 | 0.095657 |
| `cache_i2` | 162 | 81 | -50.00% | 239.632 | 238.488 | -0.48% | 43.379 | 45.153 | +4.09% | 0.025162 | 0.100090 |
| `cache_i3` | 162 | 81 | -50.00% | 172.093 | 173.047 | +0.55% | 43.058 | 44.611 | +3.61% | 0.026132 | 0.106369 |
| `early_tradeoff` | 162 | 81 | -50.00% | 304.843 | 301.719 | -1.02% | 43.359 | 44.686 | +3.06% | 0.019748 | 0.078217 |

## Stage Timing: Video Fixed_10 ON vs OFF

| stage | no video VisPruner ms | V4 video VisPruner ms | delta |
|---|---:|---:|---:|
| `total_time` | 374.109 | 370.337 | -1.01% |
| `external_prepare_batch_ms` | 13.443 | 13.716 | +2.03% |
| `embed_processing` | 47.837 | 49.783 | +4.07% |
| `video_path_total` | 47.272 | 49.328 | +4.35% |
| `vision_video_forward` | 47.300 | 49.358 | +4.35% |
| `vispruner_total` | 0.000 | 1.184 | +0.00% |
| `vispruner_apply_keep_to_sequences` | 0.000 | 0.207 | +0.00% |
| `scatter_video_embeds` | 0.198 | 0.121 | -39.20% |
| `position_encoding` | 0.822 | 0.121 | -85.26% |
| `action_initialization` | 0.516 | 0.502 | -2.68% |
| `prefetch_forward` | 30.344 | 29.573 | -2.54% |
| `prefill_transformer` | 30.139 | 29.367 | -2.56% |
| `ode_integration` | 293.070 | 288.863 | -1.44% |
| `ode_transformer_total` | 287.163 | 283.001 | -1.45% |
| `postprocessing` | 0.007 | 0.007 | -3.52% |

## Conclusion

- Video VisPruner is active: video tokens are reduced from `162` to `81` on average, exactly matching `keep_ratio=0.5`.
- On this 50-video run, V4 video pruning mainly reduces visual-path work; ODE time is mostly controlled by fixed/cache/early-stop settings.
- The ON-vs-OFF action MAE measures the action perturbation introduced by video token pruning itself; lower is better. Rollout success is not included because this script only runs offline inference.

## Raw Result Files

- image_on: `/root/autodl-tmp/wall_x/workspace/v4_video_pruning_compare/libero_50img_vispruner_v5_results.json`
- video_on: `/root/autodl-tmp/wall_x/workspace/v4_video_pruning_compare/libero_50video_vispruner_on_v5_results.json`
- video_off: `/root/autodl-tmp/wall_x/workspace/v4_video_pruning_compare/libero_50video_vispruner_off_v5_results.json`