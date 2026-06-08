# Wall-X predictor_early 筛选比例扫描完整报告（修正版）

## 报告重点

本报告重点突出两个指标：

1. **前半部分 token 处理链路耗时**：使用 `front_chain_total_time`，并展开完整 13 段时间戳。速度主结论采用稳定区间统计，即每组 300 张高分辨率 LeRobot 样本中去掉前 50 张后的 250 张均值。
2. **最终动作输出一致性**：比较 predictor_early 剪枝版本最终 Wall-X 动作输出与 original 原始版本动作输出之间的 MAE、RMSE、max_abs、cosine similarity。这里衡量的是“与原始模型输出的一致性”，不是与真实数据集 action label 的任务准确率。

## 实验设置

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- 样本数: `300，速度稳定统计使用后 250 张`
- image_min_pixels: `254016，原始视觉 token 为 324`
- pruned_strategy: `predictor_early`
- predictor_checkpoint: `/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt`
- predictor_source / layer: `early_hidden / 8`
- keep_ratios: `0.7, 0.6, 0.5, 0.4, 0.3`

## 核心结果总表

原始版本稳定前链路均值为 `131.222 ms`，中位数为 `129.989 ms`。

| keep_ratio | token | 前链路均值 ms | 耗时变化 ms | 耗时变化比例 | 前链路中位数 ms | MAE | RMSE | mean max_abs | worst max_abs | cosine | allclose_rate | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.7 | 324->227 | 710.973 | +579.752 | +441.81% | 707.340 | 0.018815 | 0.023514 | 0.075728 | 0.149651 | 0.999977 | 0.0000 | 异常路径，暂不作为候选 |
| 0.6 | 324->195 | 680.155 | +548.933 | +418.32% | 682.986 | 0.021970 | 0.027487 | 0.088868 | 0.227585 | 0.999968 | 0.0000 | 异常路径，暂不作为候选 |
| 0.5 | 324->162 | 101.340 | -29.882 | -22.77% | 95.756 | 0.025359 | 0.031705 | 0.100451 | 0.203187 | 0.999958 | 0.0000 | 有效加速，保守方案 |
| 0.4 | 324->130 | 96.300 | -34.922 | -26.61% | 91.137 | 0.028883 | 0.036149 | 0.113577 | 0.239101 | 0.999945 | 0.0000 | 推荐平衡点 |
| 0.3 | 324->98 | 90.508 | -40.714 | -31.03% | 90.103 | 0.033086 | 0.041473 | 0.128427 | 0.256607 | 0.999928 | 0.0000 | 最快，偏差更大 |

## 全量均值诊断表

该表用于解释异常点。速度主结论仍以前面的稳定区间表为准。

| keep_ratio | 全量均值 ms | 全量中位数 ms | 稳定均值 ms | >500ms 样本数 | >120ms 样本数 | 说明 |
|---:|---:|---:|---:|---:|---:|---|
| 0.7 | 707.515 | 705.742 | 710.973 | 300 | 300 | early-prune vision path 异常慢，需单独排查 |
| 0.6 | 681.509 | 684.856 | 680.155 | 300 | 300 | early-prune vision path 异常慢，需单独排查 |
| 0.5 | 180.719 | 96.069 | 101.340 | 39 | 64 | 前段异常拉高全量均值，稳定区间正常 |
| 0.4 | 96.263 | 91.124 | 96.300 | 0 | 21 | 稳定 |
| 0.3 | 90.506 | 90.115 | 90.508 | 0 | 0 | 稳定 |

## 13 个时间戳含义

| 顺序 | 时间戳 | 中文含义 | 解释 |
|---:|---|---|---|
| 1 | `s01_image_cast` | 图像 dtype 转换 | 将 pixel_values 转成 vision tower 使用的数据类型。 |
| 2 | `s02_vision_encode_or_prune` | 视觉编码/早期剪枝主段 | 原始版本完整跑 vision tower；predictor_early 在第 8 层用轻量 scorer 选 token，并只让保留 token 继续走后半段。 |
| 3 | `s03_pruning_position_prepare` | 剪枝前位置准备 | 为后续序列裁剪准备 position_ids；原始版本为 0。 |
| 4 | `s04_apply_pruning` | 应用剪枝 | 根据 keep mask 裁剪 image_embeds、input_ids、attention_mask 等序列张量。 |
| 5 | `s05_embed_tokens` | 文本/占位 token embedding | 把 input_ids 转成语言模型可用的 token embedding。 |
| 6 | `s06_scatter_image_embeds` | 写入视觉 embedding | 把裁剪后的 image_embeds 写入图像 token 对应位置。 |
| 7 | `s07_scatter_proprioception` | 写入机器人状态 embedding | 把 proprioception 投影后的 embedding 写入状态 token 位置。 |
| 8 | `s08_attention_mask_to_device` | attention mask 设备对齐 | 将 attention_mask 移到模型运行设备。 |
| 9 | `s09_position_and_moe_index` | 位置编码与 MoE 分组 | 准备 RoPE position ids 和 MoE token 分组索引。 |
| 10 | `s10_action_initialization` | 动作初始化 | 生成初始 noisy action、时间步和 action token embedding。 |
| 11 | `s11_prefill_transformer` | 主模型 prefill Transformer | 完整前缀序列首次进入主 Transformer，生成 hidden states 和 KV cache。 |
| 12 | `s12_prefill_action_head` | prefill action head | 基于 prefill 输出进行第一次动作头投影。 |
| 13 | `s13_unattributed_framework_overhead` | 未归因框架开销 | 显式时间戳之间的小差额，用于让 13 段相加等于 `front_chain_total_time`。 |
| total | `front_chain_total_time` | 前链路总耗时 | 从模型前半 token 处理链路开始，到 prefill action head 完成的总耗时；本报告的核心耗时指标。 |

## 五组比例完整 13 段时间戳对比

以下每组均包含完整 13 段时间戳和总耗时。`原始版本 ms` 使用同一批样本的 original 稳定区间均值；`剪枝版本 ms` 使用对应 keep_ratio 的 predictor_early 稳定区间均值。

### keep_ratio = 0.7

- token: `324->227`
- 前链路耗时: `131.222 ms -> 710.973 ms`，变化 `+579.752 ms (+441.81%)`
- 动作输出一致性: MAE `0.018815`，RMSE `0.023514`，mean max_abs `0.075728`，worst max_abs `0.149651`，cosine `0.999977`，allclose_rate `0.0000`

| 顺序 | 时间戳 | 中文含义 | 原始版本 ms | 剪枝版本 ms | 变化 ms | 变化比例 | 方向 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 图像 dtype 转换 | 0.050 | 1.528 | +1.478 | +2951.35% | 增加 |
| 2 | `s02_vision_encode_or_prune` | 视觉编码/早期剪枝主段 | 96.726 | 621.369 | +524.643 | +542.40% | 增加 |
| 3 | `s03_pruning_position_prepare` | 剪枝前位置准备 | 0.000 | 11.456 | +11.456 | +0.00% | 增加 |
| 4 | `s04_apply_pruning` | 应用剪枝 | 0.000 | 11.994 | +11.994 | +0.00% | 增加 |
| 5 | `s05_embed_tokens` | 文本/占位 token embedding | 0.055 | 0.016 | -0.039 | -70.88% | 减少 |
| 6 | `s06_scatter_image_embeds` | 写入视觉 embedding | 0.152 | 0.802 | +0.650 | +427.48% | 增加 |
| 7 | `s07_scatter_proprioception` | 写入机器人状态 embedding | 0.157 | 0.846 | +0.689 | +439.69% | 增加 |
| 8 | `s08_attention_mask_to_device` | attention mask 设备对齐 | 0.007 | 0.003 | -0.004 | -63.81% | 减少 |
| 9 | `s09_position_and_moe_index` | 位置编码与 MoE 分组 | 0.885 | 1.374 | +0.489 | +55.23% | 增加 |
| 10 | `s10_action_initialization` | 动作初始化 | 0.502 | 3.862 | +3.360 | +668.86% | 增加 |
| 11 | `s11_prefill_transformer` | 主模型 prefill Transformer | 32.253 | 40.995 | +8.743 | +27.11% | 增加 |
| 12 | `s12_prefill_action_head` | prefill action head | 0.128 | 1.527 | +1.399 | +1091.26% | 增加 |
| 13 | `s13_unattributed_framework_overhead` | 未归因框架开销 | 0.306 | 15.200 | +14.894 | +4860.91% | 增加 |
| total | `front_chain_total_time` | 前链路总耗时 | 131.222 | 710.973 | +579.752 | +441.81% | 增加 |

### keep_ratio = 0.6

- token: `324->195`
- 前链路耗时: `131.222 ms -> 680.155 ms`，变化 `+548.933 ms (+418.32%)`
- 动作输出一致性: MAE `0.021970`，RMSE `0.027487`，mean max_abs `0.088868`，worst max_abs `0.227585`，cosine `0.999968`，allclose_rate `0.0000`

| 顺序 | 时间戳 | 中文含义 | 原始版本 ms | 剪枝版本 ms | 变化 ms | 变化比例 | 方向 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 图像 dtype 转换 | 0.050 | 1.309 | +1.259 | +2515.09% | 增加 |
| 2 | `s02_vision_encode_or_prune` | 视觉编码/早期剪枝主段 | 96.726 | 591.167 | +494.441 | +511.18% | 增加 |
| 3 | `s03_pruning_position_prepare` | 剪枝前位置准备 | 0.000 | 9.654 | +9.654 | +0.00% | 增加 |
| 4 | `s04_apply_pruning` | 应用剪枝 | 0.000 | 12.201 | +12.201 | +0.00% | 增加 |
| 5 | `s05_embed_tokens` | 文本/占位 token embedding | 0.055 | 0.021 | -0.035 | -62.76% | 减少 |
| 6 | `s06_scatter_image_embeds` | 写入视觉 embedding | 0.152 | 0.873 | +0.721 | +473.85% | 增加 |
| 7 | `s07_scatter_proprioception` | 写入机器人状态 embedding | 0.157 | 0.846 | +0.689 | +439.57% | 增加 |
| 8 | `s08_attention_mask_to_device` | attention mask 设备对齐 | 0.007 | 0.003 | -0.004 | -58.80% | 减少 |
| 9 | `s09_position_and_moe_index` | 位置编码与 MoE 分组 | 0.885 | 1.618 | +0.733 | +82.85% | 增加 |
| 10 | `s10_action_initialization` | 动作初始化 | 0.502 | 4.473 | +3.970 | +790.36% | 增加 |
| 11 | `s11_prefill_transformer` | 主模型 prefill Transformer | 32.253 | 40.739 | +8.486 | +26.31% | 增加 |
| 12 | `s12_prefill_action_head` | prefill action head | 0.128 | 1.682 | +1.554 | +1212.20% | 增加 |
| 13 | `s13_unattributed_framework_overhead` | 未归因框架开销 | 0.306 | 15.570 | +15.263 | +4981.38% | 增加 |
| total | `front_chain_total_time` | 前链路总耗时 | 131.222 | 680.155 | +548.933 | +418.32% | 增加 |

### keep_ratio = 0.5

- token: `324->162`
- 前链路耗时: `131.222 ms -> 101.340 ms`，变化 `-29.882 ms (-22.77%)`
- 动作输出一致性: MAE `0.025359`，RMSE `0.031705`，mean max_abs `0.100451`，worst max_abs `0.203187`，cosine `0.999958`，allclose_rate `0.0000`

| 顺序 | 时间戳 | 中文含义 | 原始版本 ms | 剪枝版本 ms | 变化 ms | 变化比例 | 方向 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 图像 dtype 转换 | 0.050 | 0.049 | -0.001 | -2.39% | 减少 |
| 2 | `s02_vision_encode_or_prune` | 视觉编码/早期剪枝主段 | 96.726 | 63.046 | -33.680 | -34.82% | 减少 |
| 3 | `s03_pruning_position_prepare` | 剪枝前位置准备 | 0.000 | 0.755 | +0.755 | +0.00% | 增加 |
| 4 | `s04_apply_pruning` | 应用剪枝 | 0.000 | 0.667 | +0.667 | +0.00% | 增加 |
| 5 | `s05_embed_tokens` | 文本/占位 token embedding | 0.055 | 0.048 | -0.007 | -13.36% | 减少 |
| 6 | `s06_scatter_image_embeds` | 写入视觉 embedding | 0.152 | 0.106 | -0.046 | -30.02% | 减少 |
| 7 | `s07_scatter_proprioception` | 写入机器人状态 embedding | 0.157 | 0.132 | -0.025 | -15.94% | 减少 |
| 8 | `s08_attention_mask_to_device` | attention mask 设备对齐 | 0.007 | 0.006 | -0.001 | -8.94% | 减少 |
| 9 | `s09_position_and_moe_index` | 位置编码与 MoE 分组 | 0.885 | 0.119 | -0.766 | -86.53% | 减少 |
| 10 | `s10_action_initialization` | 动作初始化 | 0.502 | 0.481 | -0.021 | -4.20% | 减少 |
| 11 | `s11_prefill_transformer` | 主模型 prefill Transformer | 32.253 | 35.422 | +3.170 | +9.83% | 增加 |
| 12 | `s12_prefill_action_head` | prefill action head | 0.128 | 0.124 | -0.004 | -2.97% | 减少 |
| 13 | `s13_unattributed_framework_overhead` | 未归因框架开销 | 0.306 | 0.383 | +0.077 | +25.07% | 增加 |
| total | `front_chain_total_time` | 前链路总耗时 | 131.222 | 101.340 | -29.882 | -22.77% | 减少 |

### keep_ratio = 0.4

- token: `324->130`
- 前链路耗时: `131.222 ms -> 96.300 ms`，变化 `-34.922 ms (-26.61%)`
- 动作输出一致性: MAE `0.028883`，RMSE `0.036149`，mean max_abs `0.113577`，worst max_abs `0.239101`，cosine `0.999945`，allclose_rate `0.0000`

| 顺序 | 时间戳 | 中文含义 | 原始版本 ms | 剪枝版本 ms | 变化 ms | 变化比例 | 方向 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 图像 dtype 转换 | 0.050 | 0.048 | -0.002 | -4.09% | 减少 |
| 2 | `s02_vision_encode_or_prune` | 视觉编码/早期剪枝主段 | 96.726 | 58.990 | -37.736 | -39.01% | 减少 |
| 3 | `s03_pruning_position_prepare` | 剪枝前位置准备 | 0.000 | 0.720 | +0.720 | +0.00% | 增加 |
| 4 | `s04_apply_pruning` | 应用剪枝 | 0.000 | 0.661 | +0.661 | +0.00% | 增加 |
| 5 | `s05_embed_tokens` | 文本/占位 token embedding | 0.055 | 0.046 | -0.010 | -17.43% | 减少 |
| 6 | `s06_scatter_image_embeds` | 写入视觉 embedding | 0.152 | 0.105 | -0.047 | -30.70% | 减少 |
| 7 | `s07_scatter_proprioception` | 写入机器人状态 embedding | 0.157 | 0.132 | -0.025 | -16.06% | 减少 |
| 8 | `s08_attention_mask_to_device` | attention mask 设备对齐 | 0.007 | 0.006 | -0.001 | -9.65% | 减少 |
| 9 | `s09_position_and_moe_index` | 位置编码与 MoE 分组 | 0.885 | 0.119 | -0.766 | -86.58% | 减少 |
| 10 | `s10_action_initialization` | 动作初始化 | 0.502 | 0.471 | -0.031 | -6.24% | 减少 |
| 11 | `s11_prefill_transformer` | 主模型 prefill Transformer | 32.253 | 34.500 | +2.247 | +6.97% | 增加 |
| 12 | `s12_prefill_action_head` | prefill action head | 0.128 | 0.120 | -0.008 | -6.04% | 减少 |
| 13 | `s13_unattributed_framework_overhead` | 未归因框架开销 | 0.306 | 0.383 | +0.076 | +24.94% | 增加 |
| total | `front_chain_total_time` | 前链路总耗时 | 131.222 | 96.300 | -34.922 | -26.61% | 减少 |

### keep_ratio = 0.3

- token: `324->98`
- 前链路耗时: `131.222 ms -> 90.508 ms`，变化 `-40.714 ms (-31.03%)`
- 动作输出一致性: MAE `0.033086`，RMSE `0.041473`，mean max_abs `0.128427`，worst max_abs `0.256607`，cosine `0.999928`，allclose_rate `0.0000`

| 顺序 | 时间戳 | 中文含义 | 原始版本 ms | 剪枝版本 ms | 变化 ms | 变化比例 | 方向 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `s01_image_cast` | 图像 dtype 转换 | 0.050 | 0.048 | -0.002 | -3.18% | 减少 |
| 2 | `s02_vision_encode_or_prune` | 视觉编码/早期剪枝主段 | 96.726 | 58.081 | -38.645 | -39.95% | 减少 |
| 3 | `s03_pruning_position_prepare` | 剪枝前位置准备 | 0.000 | 0.745 | +0.745 | +0.00% | 增加 |
| 4 | `s04_apply_pruning` | 应用剪枝 | 0.000 | 0.658 | +0.658 | +0.00% | 增加 |
| 5 | `s05_embed_tokens` | 文本/占位 token embedding | 0.055 | 0.048 | -0.008 | -13.96% | 减少 |
| 6 | `s06_scatter_image_embeds` | 写入视觉 embedding | 0.152 | 0.104 | -0.048 | -31.35% | 减少 |
| 7 | `s07_scatter_proprioception` | 写入机器人状态 embedding | 0.157 | 0.131 | -0.025 | -16.13% | 减少 |
| 8 | `s08_attention_mask_to_device` | attention mask 设备对齐 | 0.007 | 0.006 | -0.001 | -10.13% | 减少 |
| 9 | `s09_position_and_moe_index` | 位置编码与 MoE 分组 | 0.885 | 0.118 | -0.767 | -86.62% | 减少 |
| 10 | `s10_action_initialization` | 动作初始化 | 0.502 | 0.474 | -0.028 | -5.63% | 减少 |
| 11 | `s11_prefill_transformer` | 主模型 prefill Transformer | 32.253 | 29.586 | -2.667 | -8.27% | 减少 |
| 12 | `s12_prefill_action_head` | prefill action head | 0.128 | 0.119 | -0.009 | -7.17% | 减少 |
| 13 | `s13_unattributed_framework_overhead` | 未归因框架开销 | 0.306 | 0.387 | +0.081 | +26.32% | 增加 |
| total | `front_chain_total_time` | 前链路总耗时 | 131.222 | 90.508 | -40.714 | -31.03% | 减少 |

## 0.7 / 0.6 异常说明

`keep_ratio=0.7` 和 `keep_ratio=0.6` 的 400% 以上耗时增加主要集中在 `s02_vision_encode_or_prune`，也就是 predictor_early 的 vision tower 内部 early-prune 编码阶段。它们不应解释为正常剪枝比例规律，也不建议作为正式候选比例。

稳定区间中，原始版本 `s02` 约为 `96.726 ms`；`0.7` 的 `s02` 约为 `621.369 ms`，`0.6` 的 `s02` 约为 `591.167 ms`。相比之下，`0.5` 的 `s02` 约为 `63.046 ms`，`0.4/0.3` 约为 `59 ms / 58 ms`。这说明异常主要来自 early-prune vision path 内部，可能与动态裁剪后的变长 attention shape 或底层 kernel fallback 有关。

## 动作输出一致性说明

报告中的动作指标不是任务成功率，也不是和真实 action label 的准确率。它衡量的是剪枝版本最终输出动作与 original 原始版本输出动作的数值接近程度。cosine similarity 很高说明整体动作方向接近；但 allclose_rate 全部为 0，说明剪枝后动作并不是逐元素完全相同。MAE/RMSE 随 keep_ratio 降低而上升，这符合 token 保留越少、输出偏差越大的趋势。

## 推荐结论

`keep_ratio=0.4` 是当前推荐平衡点：前链路稳定加速 `26.61%`，动作输出一致性仍较高。`keep_ratio=0.5` 是保守有效方案，稳定加速 `22.77%`。`keep_ratio=0.3` 速度最快，稳定加速 `31.03%`，但动作偏差更大，需要结合真实机器人任务验证。`0.7/0.6` 暂不作为候选方案，建议作为工程异常单独排查。

## 原始结果文件

- JSON 原始结果：`/root/autodl-tmp/wall_x/wall_x/report/mid_exam_300_2_keep_ratio_sweep_results.json`
