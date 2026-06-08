# VisPruner 打分器第 8 层应用选择分析报告

## 1. 结论

当前选择 `predictor_early_layer=8` 不是随意指定，而是一个工程上较合理、并已被现有实验验证过的折中点。它的核心价值是：**在 vision tower 已完成一次全局视觉信息混合之后尽早剪枝，从而保留足够的 token 判断质量，同时让后续大部分 vision tower 层和主模型前链路享受 token 减少带来的收益。**

需要特别澄清：根据当前模型配置，Wall-X 使用的视觉塔 `vision_config.depth=32`，不是 16 层。第 8 层是视觉塔的 1/4 深度位置，并且刚好位于第一个 full-attention block 之后。

因此，第 8 层的定位可以概括为：

| 维度 | 第 8 层的意义 |
|---|---|
| 语义质量 | 已经过 8 个 vision block，并经过第一个 full-attention block，token feature 不再只是低层局部纹理。 |
| 计算收益 | 仍有后续 24 层 vision tower 可以在剪枝后少算 token，保留大部分 early-prune 加速空间。 |
| 训练一致性 | 当前 teacher score、scorer checkpoint、评估结果全部基于 `early_hidden + layer 8`。 |
| 工程稳定性 | 代码默认逻辑本身就是 `len(self.blocks)//4`，在当前 depth=32 下等于 8。 |

## 2. 代码依据

### 2.1 模型配置：vision tower 是 32 层

当前模型配置文件 `pretrained/wall-oss-fast/config.json` 中：

```json
"vision_config": {
  "depth": 32,
  "hidden_size": 1280,
  "spatial_merge_size": 2,
  "fullatt_block_indexes": [7, 15, 23, 31]
}
```

这说明视觉塔共有 32 个 block，full-attention block 位于 0-based index `7/15/23/31`，对应第 `8/16/24/32` 层。

### 2.2 第 8 层正好在第一个 full-attention block 之后

在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py` 中，early-prune 触发条件是：

```python
if (
    vispruner_early_prune
    and vispruner_feature_source == "early_hidden"
    and vispruner_keep_mask is None
    and layer_num + 1 >= early_layer
):
    early_features = hidden_states.reshape(...)
    vispruner_scores = scorer(early_features)
    vispruner_keep_mask = self._vispruner_keep_mask_from_scores(...)
    hidden_states = hidden_states[keep_mask_window]
```

因此，当 `early_layer=8` 时，剪枝发生在 `layer_num=7` 的 block 运行之后。由于 `fullatt_block_indexes` 包含 `7`，这意味着 scorer 看到的是**经过第一次 full-attention 全局交互后的视觉 token hidden feature**。

这比更早层更稳：如果在第 4/6 层就剪枝，token feature 主要还是局部/window attention 后的低层视觉特征，可能更偏纹理、边缘和局部区域，未必能稳定判断哪些 token 对最终动作推理重要。

### 2.3 代码默认值也指向 1/4 深度

同一文件中 early layer 的默认逻辑是：

```python
early_layer = (
    int(vispruner_early_layer)
    if vispruner_early_layer is not None
    else max(1, len(self.blocks) // 4)
)
```

在当前 `len(self.blocks)=32` 时，默认 `len(self.blocks)//4=8`。这说明从实现设计上，1/4 视觉塔深度就是一个被认为合理的早期剪枝位置。

### 2.4 推理路径实际使用第 8 层

在 `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py` 中，`predictor_early` 会把配置传给 visual tower：

```python
image_embeds, image_scores, image_keep_mask = self.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    output_attentions=False,
    vispruner_early_prune=True,
    vispruner_score_predictor=self.vispruner_score_predictor,
    vispruner_feature_source="early_hidden",
    vispruner_early_layer=self.config.vispruner_predictor_early_layer,
    vispruner_keep_ratio=self.config.vispruner_keep_ratio,
)
```

配置读取路径在 `wall_x/model/model_utils.py` 中，`early_layer` 会写入 `model_config.vispruner_predictor_early_layer`，因此训练配置、推理配置和模型 forward 是一致的。

## 3. 训练依据

当前最终使用的 scorer checkpoint 是：

```text
workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt
```

这个命名中的 `early_l8` 表示 teacher feature 和 scorer 输入都来自第 8 层 early hidden。已有技术文档记录的训练设置如下：

| 项目 | 数值 |
|---|---:|
| 数据集 | `libero_all` |
| 训练图片数 | `30000` |
| token 级样本数 | `2430000` |
| feature_source | `early_hidden` |
| early_layer | `8` |
| keep_ratio | `0.5` |
| 模型结构 | `1280 -> 320 -> 1` MLP |
| loss | MSE 回归 teacher score |
| best_loss | `0.062204` |
| mean_topk_overlap | `0.922611` |
| mean_mask_agreement | `0.921656` |

这说明第 8 层 feature 对 topk_attention teacher 的 token 选择具有较强可预测性：scorer 约 92% 地复现了 teacher 的保留 token 结果。

也正因为 checkpoint 是在第 8 层 feature 上训练的，后续如果把 scorer 移到第 4/6/10/16 层，不能直接复用当前 checkpoint。不同层 hidden state 的分布、语义含量和 token 排序都会变化，需要重新收集 teacher feature 并重新训练/评估 scorer。

## 4. 速度实验依据

### 4.1 第 8 层 predictor_early 能真正减少前链路耗时

在高分辨率 324-token 输入上，`predictor_early_layer=8` 的 0.5 keep_ratio 已经多次显示出前链路收益。

| 实验 | 样本数 | token | original front_chain | predictor_early layer8 | 变化 |
|---|---:|---:|---:|---:|---:|
| `front_200_324tok_predictor_early_l8` | 200 | 324 -> 162 | 130.350 ms | 95.543 ms | -34.807 ms (-26.70%) |
| `mid_exam_300_1_front_chain` | 300 | 324 -> 162 | 132.071 ms | 98.886 ms | -33.186 ms (-25.13%) |
| `mid_exam_300_2` 稳定区间 | 250 | 324 -> 162 | 131.222 ms | 101.340 ms | -29.882 ms (-22.77%) |

其中 `s02_vision_encode_or_prune` 是最关键的收益来源：

| 实验 | original s02 | predictor_early layer8 s02 | 变化 |
|---|---:|---:|---:|
| `front_200_324tok_predictor_early_l8` | 96.586 ms | 62.234 ms | -34.352 ms |
| `mid_exam_300_1_front_chain` | 98.171 ms | 64.365 ms | -33.807 ms |
| `mid_exam_300_2` 0.5 稳定区间 | 96.726 ms | 63.046 ms | -33.680 ms |

这证明第 8 层剪枝不是只减少主模型输入 token，而是已经让 vision tower 后续路径少处理 token，从而真正降低前半 token 处理链路耗时。

### 4.2 相比第一版 topk_attention，第 8 层 early pruning 才能节省 vision tower 后半段

在 200 张 324-token 高分辨率样本中，第一版 `topk_attention` 也能把 token 从 324 剪到 162，但它发生在完整 vision tower 之后，因此前链路没有加速：

| 方案 | token | original front_chain | pruned front_chain | 变化 |
|---|---:|---:|---:|---:|
| topk_attention | 324 -> 162 | 129.303 ms | 131.498 ms | +2.195 ms (+1.70%) |
| predictor_early layer8 | 324 -> 162 | 130.350 ms | 95.543 ms | -34.807 ms (-26.70%) |

这正是选择 early layer 的原因：如果等完整 vision tower 结束再打分，视觉 token 少了，但视觉编码成本已经付完；如果在第 8 层打分，后续 24 层可以少算 token。

## 5. 动作输出一致性依据

第 8 层方案不仅加速前链路，还保持了与 original 输出动作的较高一致性。以 300 张高分辨率样本、keep_ratio=0.5 为例：

| 指标 | 数值 | 含义 |
|---|---:|---|
| MAE | `0.025359` | 剪枝输出动作与 original 输出动作的平均绝对差异 |
| RMSE | `0.031705` | 均方根误差 |
| mean max_abs | `0.100451` | 每个样本最大元素误差的平均值 |
| worst max_abs | `0.203187` | 300 张样本中最坏的单元素误差 |
| cosine similarity | `0.999958` | 展平动作向量方向高度一致 |
| allclose_rate | `0.0000` | 不代表完全相同；说明逐元素严格 allclose 并未通过 |

这里的指标应称为“动作输出一致性”，不是任务成功率，也不是与真实 action label 的准确率。它说明第 8 层 scorer 在保留速度收益的同时，没有让最终动作输出发生大方向偏移。

## 6. 为什么不是更早层

更早层例如第 4/6 层理论上能节省更多 vision tower 计算，但当前不作为默认选择，原因是：

1. 第 4/6 层还没有经过第一个 full-attention block，token feature 更偏局部纹理和窗口内信息，缺少全局对象/场景交互。
2. 轻量 scorer 的 teacher 是 topk_attention 的 token importance。越早层的 feature 和 teacher score 的语义距离越远，可能导致 top-k overlap 降低。
3. 当前没有第 4/6 层的 30000 样本 scorer checkpoint，也没有相同规模的动作一致性和前链路对比结果。直接切层会导致训练/推理分布不一致。

因此，更早层是值得后续探索的“潜在更快方案”，但不是当前最稳的报告版本。

## 7. 为什么不是更晚层

更晚层例如第 16/24 层可能具有更强语义表达，但会显著减少 early pruning 的计算收益：

1. 第 16 层后才剪枝，只能节省后 16 层 vision tower；第 8 层后剪枝可以节省后 24 层。
2. 越晚越接近 `predictor_score` 或完整 vision tower 后打分，视觉编码成本已经付出更多，端到端前链路收益会下降。
3. 当前研究目标是验证“视觉 token 减少能否加快前半 token 处理链路”，所以需要一个足够早、但又不太早的位置。第 8 层正好满足这个目标。

## 8. 当前结论的边界

需要明确：现有数据证明第 8 层是当前合理且已验证的工程折中点，但还不能严格证明它是全局最优层。更严谨的层数选择实验应当重新训练并比较：

```text
early_layer = 4 / 6 / 8 / 10 / 16
```

每个层数都需要独立收集 teacher feature、训练 scorer，并在同一批高分辨率样本上比较：

| 必测指标 | 说明 |
|---|---|
| top-k overlap / mask agreement | scorer 是否能复现 teacher token 选择 |
| front_chain_total_time | 前半 token 处理链路是否加速 |
| s02_vision_encode_or_prune | vision tower early-prune 主段是否下降 |
| 动作输出一致性 | MAE/RMSE/max_abs/cosine 是否可接受 |
| 真实任务指标 | 与 action label 的误差或机器人任务成功率 |

## 9. 最终判断

选择第 8 层的理由可以总结为：

1. **代码结构合理**：32 层 vision tower 的 1/4 深度，且位于第一个 full-attention block 之后。
2. **训练结果可靠**：30k 图片、243 万 token 级样本训练后，top-k overlap 和 mask agreement 都约为 92%。
3. **速度收益明确**：高分辨率 324-token 场景下，keep_ratio=0.5 前链路稳定加速约 22%-27%。
4. **动作一致性较高**：keep_ratio=0.5 时 cosine similarity 为 0.999958，MAE 为 0.025359。
5. **工程目标匹配**：它比完整 vision tower 后打分更早，能节省后 24 层计算；又比第 4/6 层更稳，因为已经经过一次全局 attention。

所以，第 8 层是当前阶段最合理的默认应用层数。后续若要追求进一步加速，可以把第 4/6 层作为下一阶段研究方向，但必须重新训练 scorer 并重新验证动作一致性。
