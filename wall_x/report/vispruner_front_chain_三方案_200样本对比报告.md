# Wall-X 视觉 Token 剪枝前链路加速三方案对比报告

测试日期：2026-05-29；每组 200 张官方数据集图片；不包含 ODE 动作推理。

## 减少收益和抵消开销拆分

| 场景 | 方案 | token | 减少的环节 | 减少数值 | 增加/抵消项 | 完整前链路净变化 |
|---|---|---:|---|---:|---|---:|
| 普通官方样本 | topk_attention | 81 -> 41 | 下游 token 链路减少 | -1.258 ms (-4.17%) | image_path_total 增加 +1.536 ms (+5.68%) | 完整前链路 +0.278 ms (+0.49%) |
| 高分辨率样本 | topk_attention | 324 -> 162 | position_encoding 减少 | -0.579 ms (-83.30%) | image_path_total 增加 +1.855 ms (+1.92%)；下游合计 +0.329 ms (+1.04%) | 完整前链路 +2.184 ms (+1.70%) |
| 高分辨率样本 | predictor_early | 324 -> 162 | image_path_total 减少；下游 token 链路减少 | -33.010 ms (-34.14%)；-1.640 ms (-5.04%) | 剪枝准备等已计入净值 | 完整前链路 -34.650 ms (-26.81%) |

## 口径说明

- 完整前链路 direct = image_path_total + scatter_image_embeds + position_encoding + prefill_transformer。
- 剪枝后下游 token 链路 = scatter_image_embeds + position_encoding + prefill_transformer。

## 详细数据

### 普通官方样本：original vs topk_attention
| 指标 | key | original | topk_attention | 变化 |
|---|---|---:|---:|---:|
| 剪枝后下游 token 链路 | `downstream_token_ms` | 30.176 ms | 28.918 ms | -1.258 ms (-4.17%) |
| 完整前链路 direct 合计 | `front_direct_ms` | 57.209 ms | 57.487 ms | +0.278 ms (+0.49%) |
| 完整前链路 top-level 合计 | `front_top_level_ms` | 57.648 ms | 57.921 ms | +0.272 ms (+0.47%) |
| 本脚本前链路 total_time | `total_time` | 58.148 ms | 58.424 ms | +0.276 ms (+0.47%) |
| 模型外输入准备 | `external_prepare_batch_ms` | 3.725 ms | 3.749 ms | +0.024 ms (+0.64%) |
| 图像路径/视觉编码总段 | `image_path_total` | 27.033 ms | 28.569 ms | +1.536 ms (+5.68%) |
| 普通视觉编码 | `vision_image_encode` | 26.943 ms | 0.000 ms | -26.943 ms (-100.00%) |
| 完整视觉编码+attention 分数 | `vision_image_encode_score` | 0.000 ms | 27.045 ms | +27.045 ms (+0.00%) |
| VisPruner 裁剪总段 | `vispruner_total` | 0.000 ms | 0.830 ms | +0.830 ms (+0.00%) |
| 剪枝前 position ids 准备 | `pruning_position_ids_prepare` | 0.000 ms | 0.528 ms | +0.528 ms (+0.00%) |
| 图像 embedding 写入 | `scatter_image_embeds` | 0.112 ms | 0.097 ms | -0.016 ms (-13.96%) |
| 位置编码/MoE 索引 | `position_encoding` | 0.620 ms | 0.124 ms | -0.496 ms (-80.01%) |
| prefill Transformer | `prefill_transformer` | 29.443 ms | 28.698 ms | -0.746 ms (-2.53%) |

### 高分辨率样本：original vs topk_attention
| 指标 | key | original | topk_attention | 变化 |
|---|---|---:|---:|---:|
| 剪枝后下游 token 链路 | `downstream_token_ms` | 31.777 ms | 32.106 ms | +0.329 ms (+1.04%) |
| 完整前链路 direct 合计 | `front_direct_ms` | 128.312 ms | 130.496 ms | +2.184 ms (+1.70%) |
| 完整前链路 top-level 合计 | `front_top_level_ms` | 128.771 ms | 130.957 ms | +2.186 ms (+1.70%) |
| 本脚本前链路 total_time | `total_time` | 129.303 ms | 131.498 ms | +2.195 ms (+1.70%) |
| 模型外输入准备 | `external_prepare_batch_ms` | 7.601 ms | 8.679 ms | +1.078 ms (+14.18%) |
| 图像路径/视觉编码总段 | `image_path_total` | 96.535 ms | 98.390 ms | +1.855 ms (+1.92%) |
| 普通视觉编码 | `vision_image_encode` | 96.441 ms | 0.000 ms | -96.441 ms (-100.00%) |
| 完整视觉编码+attention 分数 | `vision_image_encode_score` | 0.000 ms | 96.619 ms | +96.619 ms (+0.00%) |
| VisPruner 裁剪总段 | `vispruner_total` | 0.000 ms | 0.907 ms | +0.907 ms (+0.00%) |
| 剪枝前 position ids 准备 | `pruning_position_ids_prepare` | 0.000 ms | 0.680 ms | +0.680 ms (+0.00%) |
| 图像 embedding 写入 | `scatter_image_embeds` | 0.123 ms | 0.624 ms | +0.502 ms (+408.02%) |
| 位置编码/MoE 索引 | `position_encoding` | 0.695 ms | 0.116 ms | -0.579 ms (-83.30%) |
| prefill Transformer | `prefill_transformer` | 30.959 ms | 31.365 ms | +0.406 ms (+1.31%) |

### 高分辨率样本：original vs predictor_early
| 指标 | key | original | predictor_early | 变化 |
|---|---|---:|---:|---:|
| 剪枝后下游 token 链路 | `downstream_token_ms` | 32.538 ms | 30.899 ms | -1.640 ms (-5.04%) |
| 完整前链路 direct 合计 | `front_direct_ms` | 129.231 ms | 94.581 ms | -34.650 ms (-26.81%) |
| 完整前链路 top-level 合计 | `front_top_level_ms` | 129.764 ms | 95.023 ms | -34.741 ms (-26.77%) |
| 本脚本前链路 total_time | `total_time` | 130.350 ms | 95.543 ms | -34.807 ms (-26.70%) |
| 模型外输入准备 | `external_prepare_batch_ms` | 7.955 ms | 7.546 ms | -0.409 ms (-5.14%) |
| 图像路径/视觉编码总段 | `image_path_total` | 96.692 ms | 63.682 ms | -33.010 ms (-34.14%) |
| 普通视觉编码 | `vision_image_encode` | 96.586 ms | 0.000 ms | -96.586 ms (-100.00%) |
| early prune 视觉编码 | `vision_image_encode_early_prune` | 0.000 ms | 62.234 ms | +62.234 ms (+0.00%) |
| VisPruner 裁剪总段 | `vispruner_total` | 0.000 ms | 0.679 ms | +0.679 ms (+0.00%) |
| 剪枝前 position ids 准备 | `pruning_position_ids_prepare` | 0.000 ms | 0.598 ms | +0.598 ms (+0.00%) |
| 图像 embedding 写入 | `scatter_image_embeds` | 0.157 ms | 0.100 ms | -0.057 ms (-36.08%) |
| 位置编码/MoE 索引 | `position_encoding` | 0.839 ms | 0.114 ms | -0.725 ms (-86.41%) |
| prefill Transformer | `prefill_transformer` | 31.543 ms | 30.685 ms | -0.858 ms (-2.72%) |

## 适配场景

- topk_attention：适合作为 teacher、稳定对照基线和低风险后置剪枝方案；普通样本下游 token 链路会变快，但完整前链路净收益可能被 attention score 成本抵消。
- predictor_early：适合高分辨率、多相机、多视角、长视觉上下文、300+ token 的机器人训练/部署场景；本次 324-token 完整前链路下降 26.81%。

## 原始文件
- workspace/vispruner_logs/front_200_normal_topk_results.json
- workspace/vispruner_logs/front_200_324tok_topk_results.json
- workspace/vispruner_logs/front_200_324tok_predictor_early_l8_results.json
- scripts/profile_vispruner_front_chain_lerobot.py
