# VisPruner 打分器实现精简说明

本文档只介绍当前轻量打分器 `TokenScorePredictor` 是怎么实现的。

## 1. 打分器做什么

打分器的任务是：给每个视觉 token 预测一个重要性分数。

```text
输入:  一个视觉 token 的 early hidden feature
输出:  这个 token 的 importance score
用途:  按 score 做 top-k，保留重要 token，丢弃不重要 token
```

当前剪枝比例：

```text
keep_ratio = 0.5
```

所以：

```text
81 token  -> 41 token
324 token -> 162 token
```

## 2. 打分器输入是什么

打分器不直接看原始图片，也不使用最终 vision embedding，而是使用 vision tower 中间层的 hidden state。

当前设置：

```text
predictor_source = early_hidden
predictor_early_layer = 8
```

也就是：图片经过 vision tower 前 8 层后，取此时每个视觉 token 的 hidden feature 作为 scorer 输入。

代码位置：

```text
wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py
```

核心代码：

```python
early_features = hidden_states.reshape(
    original_num_groups, self.spatial_merge_unit, -1
).mean(dim=1)[reverse_indices]
```

含义：

```text
hidden_states       : vision tower 第 8 层后的 patch hidden states
spatial_merge_unit  : 多个 patch 合并成一个 LLM 视觉 token
mean(dim=1)         : 将 patch-level feature 聚合成 token-level feature
reverse_indices     : 恢复视觉 token 的原始顺序
```

最终每个 token 的输入向量：

```text
x_j ∈ R^1280
```

## 3. 打分器模型结构

代码位置：

```text
wall_x/model/vispruner_score_predictor.py
```

模型是一个很小的 MLP：

```python
class TokenScorePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout=0.0):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token_features):
        return self.net(token_features).squeeze(-1)
```

当前 checkpoint 的结构：

| 参数 | 数值 |
|---|---:|
| input_dim | `1280` |
| hidden_dim | `320` |
| output_dim | `1` |
| dropout | `0.0` |
| 参数量 | `410241` |

数学形式：

```text
h_j = GELU(W1 x_j + b1)
score_j = W2 h_j + b2
```

它是逐 token 独立打分的模块，不做 attention，也不改变 token embedding。

## 4. 训练标签怎么来

打分器的 teacher 是第一版 `topk_attention`。

收集 teacher 时，完整跑 vision tower，并打开 attention：

```python
image_embeds, image_scores, predictor_features = model.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    output_attentions=True,
    return_vispruner_features=True,
    vispruner_feature_source="early_hidden",
    vispruner_early_layer=8,
)
```

其中：

```text
predictor_features : 第 8 层 early hidden，作为 scorer 输入
image_scores       : topk_attention 生成的 teacher score，作为训练标签
```

训练样本可以理解为：

```text
(early_hidden token feature, teacher attention score)
```

## 5. Loss 怎么定义

训练时对每张图内部的 teacher score 做标准化：

```python
teacher_target = record["image_scores"].float().reshape(-1)
teacher_target = (teacher_target - teacher_target.mean()) / (
    teacher_target.std(unbiased=False) + 1e-6
)
```

然后用 MSE 回归 teacher score：

```python
pred = model(xs)
loss = F.mse_loss(pred.float(), teacher_target.float())
```

选择回归 score 而不是直接学习 0/1 mask 的原因是：剪枝最终依赖 token 的排序，连续 score 比二值 mask 保留更多重要性顺序信息。

## 6. 关键训练参数

最终使用的 checkpoint：

```text
workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt
```

| 参数 | 数值 |
|---|---|
| 数据集 | `libero_all` |
| 图片字段 | `observation.images.faceImg` |
| teacher 策略 | `topk_attention` |
| 训练图片数 | `30000` |
| token 级样本数 | `2430000` |
| feature_source | `early_hidden` |
| early_layer | `8` |
| keep_ratio | `0.5` |
| target | `score` |
| loss | `MSE` |
| epochs | `20` |
| batch_size | `4096` |
| learning_rate | `1e-3` |
| weight_decay | `1e-4` |
| seed | `42` |

训练结果：

```text
best_loss = 0.062204
mean_topk_overlap = 0.922611
mean_mask_agreement = 0.921656
```

说明 scorer 约 92% 地复现了 `topk_attention` 的 token 选择结果。

## 7. 推理时怎么用

推理阶段使用：

```text
strategy = predictor_early
```

调用方式：

```python
image_embeds, image_scores, image_keep_mask = self.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    output_attentions=False,
    vispruner_early_prune=True,
    vispruner_score_predictor=self.vispruner_score_predictor,
    vispruner_feature_source="early_hidden",
    vispruner_early_layer=8,
    vispruner_keep_ratio=0.5,
)
```

在 vision tower 第 8 层后：

```python
vispruner_scores = scorer(early_features)
vispruner_keep_mask = topk(vispruner_scores, keep_ratio=0.5)
hidden_states = hidden_states[vispruner_keep_mask]
```

之后，vision tower 后半段只处理保留下来的 token。

这就是 `predictor_early` 能加速前半视觉 token 处理链路的关键。

## 8. 一句话总结

这个打分器本质上是一个 `1280 -> 320 -> 1` 的轻量 MLP。它用 vision tower 第 8 层的 token hidden feature 作为输入，学习第一版 `topk_attention` 的 teacher score，推理时提前给 token 打分并 top-k 剪枝，从而减少后续 vision tower 和主模型需要处理的视觉 token 数。
