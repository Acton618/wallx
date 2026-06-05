# VisPruner 轻量打分器实现技术文档

本文档专门说明当前 Wall-X VisPruner 轻量打分器 `TokenScorePredictor` 是如何构造、训练和在推理中使用的。重点不是项目背景，而是打分器本身的技术细节。

## 1. 打分器要解决的问题

在 VisPruner 中，我们需要给每个视觉 token 一个重要性分数：

```text
score_j = token j 的重要性
```

然后按分数选出前 `keep_ratio` 的 token：

```text
keep_count = ceil(num_visual_tokens * keep_ratio)
keep_mask = topk(score, keep_count)
```

第一版 `topk_attention` 的分数来自完整 vision tower 的 attention，因此必须先跑完整视觉编码。当前轻量打分器的目标是：在 vision tower 中间层就预测 token 分数，从而提前剪掉 token，让后半段 vision tower 少算。

当前最终使用的推理策略是：

```text
strategy = predictor_early
predictor_source = early_hidden
predictor_early_layer = 8
keep_ratio = 0.5
```

## 2. 打分器的输入是什么

打分器输入不是原始图片，也不是最终图像 embedding，而是 vision tower 第 8 层后的 token hidden state。

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

1. `hidden_states` 是 vision tower 中间层的 patch-level hidden state。
2. `spatial_merge_unit` 表示若干 patch 会合并成一个 LLM 级视觉 token。
3. `reshape(...).mean(dim=1)` 把 patch-level hidden state 聚合成 LLM 级视觉 token feature。
4. `[reverse_indices]` 把 window attention 内部重排过的 token 顺序恢复到原始图像 token 顺序。

因此，每个视觉 token 的打分器输入是：

```text
x_j ∈ R^1280
```

其中：

```text
j = 第 j 个视觉 token
1280 = 当前 vision hidden size
```

对于一张默认 LeRobot 图片：

```text
输入特征 shape: [81, 1280]
```

对于高分辨率实验图片：

```text
输入特征 shape: [324, 1280]
```

## 3. Teacher 分数从哪里来

打分器不是人工标注训练的，而是用第一版 `topk_attention` 作为 teacher。

Teacher 收集脚本：

```text
scripts/collect_vispruner_teacher_scores.py
```

收集时调用完整 vision tower，并打开 attention 输出：

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

这里同时得到：

| 变量 | 含义 |
|---|---|
| `image_embeds` | 完整 vision tower 输出后的视觉 embedding。 |
| `image_scores` | `topk_attention` teacher 生成的 token 重要性分数。 |
| `predictor_features` | 第 8 层 early hidden 特征，作为轻量打分器输入。 |

在 vision tower 内部，teacher attention score 的生成逻辑可以概括为：

```python
if output_attentions:
    hidden_states, attn_weights = block_outputs
    patch_scores = attn_weights.detach().float().mean(dim=0).mean(dim=0)
    patch_score_sum = patch_score_sum + patch_scores
    patch_score_count += 1

patch_scores = patch_score_sum / patch_score_count
merged_scores = patch_scores.reshape(
    seq_len // self.spatial_merge_unit,
    self.spatial_merge_unit
).mean(dim=1)
merged_scores = merged_scores[reverse_indices]
```

也就是说，teacher 分数是从 vision tower attention 权重聚合而来的。它先在 patch 级别统计注意力强度，再按 `spatial_merge_unit` 聚合成 LLM 级视觉 token 分数。

## 4. Teacher keep mask 怎么生成

有了 teacher score 后，仍然使用原 VisPruner 的 top-k 逻辑生成保留 mask。

代码位置：

```text
wall_x/model/vispruner_token_pruner.py
```

核心逻辑：

```python
local_scores = scores[offset : offset + length]
local_keep = torch.topk(local_scores, k=keep_count, largest=True).indices
local_keep = local_keep.sort().values
keep_mask[offset + local_keep] = True
```

其中：

```text
length = 当前图片视觉 token 数
keep_count = ceil(length * keep_ratio)
keep_ratio = 0.5
```

例如：

```text
默认样本: 81 token -> ceil(81 * 0.5) = 41 token
高分辨率样本: 324 token -> ceil(324 * 0.5) = 162 token
```

注意：`local_keep.sort()` 会把被保留 token 恢复成原始顺序。也就是说，VisPruner 只删除 token，不重新排列 token。

## 5. 训练数据长什么样

每张图片保存一个 record。核心字段如下：

```python
record = {
    "dataset_index": int(dataset_idx),
    "image_scores": image_scores.detach().float().cpu(),
    "keep_mask": keep_mask.detach().cpu(),
    "keep_indices": [item.detach().cpu() for item in keep_indices],
    "predictor_features": predictor_features,
    "predictor_feature_source": "early_hidden",
    "predictor_early_layer": 8,
}
```

训练时实际使用两项：

| 字段 | 用途 |
|---|---|
| `predictor_features` | 输入 `x_j` |
| `image_scores` | 回归目标 `y_j` |

`keep_mask` 主要用于评估 scorer 选 token 是否接近 teacher。

## 6. 打分器模型结构

代码位置：

```text
wall_x/model/vispruner_score_predictor.py
```

模型定义：

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

数学形式：

```text
h_j = GELU(W1 x_j + b1)
score_j = W2 h_j + b2
```

当前 checkpoint 的结构参数：

| 参数 | 值 |
|---|---:|
| input_dim | `1280` |
| hidden_dim | `320` |
| dropout | `0.0` |
| output_dim | `1` |
| 参数量 | `410241` |

参数量计算：

```text
Linear(1280, 320): 1280 * 320 + 320 = 409920
Linear(320, 1): 320 * 1 + 1 = 321
total = 410241
```

这个模型是逐 token 独立打分的 MLP。它不做 token-token attention，也不改变视觉 embedding，只输出每个 token 的重要性分数。

## 7. 训练目标

训练脚本：

```text
scripts/train_vispruner_score_predictor.py
```

本项目使用 `target=score`，即拟合 teacher attention score，而不是直接拟合二值 keep mask。

训练时先对每张图内部的 teacher score 做标准化：

```python
teacher_target = record["image_scores"].float().reshape(-1)
teacher_target = (teacher_target - teacher_target.mean()) / (
    teacher_target.std(unbiased=False) + 1e-6
)
```

然后用 MSE loss：

```python
pred = model(xs)
loss = F.mse_loss(pred.float(), teacher_target.float())
```

完整目标函数：

```text
L(θ) = mean_j ( f_θ(x_j) - normalize(s_teacher_j) )^2
```

其中：

```text
x_j = 第 j 个 token 的 early_hidden 特征
f_θ(x_j) = 轻量打分器预测分数
s_teacher_j = topk_attention teacher 分数
```

为什么选择 score 而不是 mask：

1. 剪枝真正依赖的是 token 分数排序。
2. 连续 score 比 0/1 mask 提供更多相对重要性信息。
3. 推理时仍然通过 top-k 把 score 转成 keep mask。

## 8. 实际训练参数

最终使用的 checkpoint：

```text
workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt
```

Teacher 数据：

```text
workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards
```

具体参数：

| 类别 | 参数 | 值 |
|---|---|---|
| 数据 | dataset_root | `/root/autodl-tmp/wall_x/datasheet/libero_all` |
| 数据 | repo_id | `libero_all` |
| 数据 | image_key | `observation.images.faceImg` |
| 数据 | teacher records | `30000` |
| 数据 | shards | `15` |
| 数据 | records per shard | `2000` |
| 数据 | feature_dtype | `float16` |
| Teacher | strategy | `topk_attention` |
| Teacher | keep_ratio | `0.5` |
| Feature | source | `early_hidden` |
| Feature | early_layer | `8` |
| Feature | input_dim | `1280` |
| Model | hidden_dim | `320` |
| Model | dropout | `0.0` |
| Train | target | `score` |
| Train | loss | `MSE` |
| Train | epochs | `20` |
| Train | batch_size | `4096` |
| Train | learning_rate | `1e-3` |
| Train | weight_decay | `1e-4` |
| Train | seed | `42` |
| Train | val_ratio | `0` |

本次训练数据来自默认 LeRobot 图像处理口径：

```text
每张图 81 个视觉 token
30000 张图 * 81 token = 2430000 个 token 级训练样本
```

训练命令：

```bash
python -u scripts/train_vispruner_score_predictor.py \
  --teacher-path /root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards \
  --output-path /root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt \
  --feature-key auto \
  --target score \
  --epochs 20 \
  --batch-size 4096 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.0 \
  --val-ratio 0 \
  --seed 42
```

## 9. 训练结果

训练日志中的关键 loss：

| 阶段 | train_loss |
|---|---:|
| epoch 1 | `0.157768` |
| epoch 2 | `0.112918` |
| epoch 3 | `0.100703` |
| epoch 4 | `0.093792` |
| epoch 20 | `0.062212` |
| checkpoint best_loss | `0.062204` |

可以看到，训练初期 loss 明显下降，后期稳定在约 `0.062`。

## 10. 评估方法

评估脚本：

```text
scripts/evaluate_vispruner_score_predictor.py
```

评估时先用 scorer 预测分数：

```python
pred_scores = predictor(features).detach().float().cpu().reshape(-1)
```

然后保留和 teacher 相同数量的 top-k token：

```python
pred_mask = make_topk_mask(pred_scores, int(teacher_mask.sum().item()))
```

三个核心指标：

```python
score_mse = F.mse_loss(pred_scores, teacher_scores)
topk_overlap = (pred_mask & teacher_mask).sum() / teacher_mask.sum()
mask_agreement = (pred_mask == teacher_mask).mean()
```

评估结果：

| 指标 | 数值 | 解释 |
|---|---:|---|
| records | `30000` | 评估图片数 |
| shards | `15` | teacher shard 数 |
| mean_score_mse | `0.076447` | 预测分数与 teacher 分数的 MSE |
| mean_topk_overlap | `0.922611` | scorer 选中 token 与 teacher 选中 token 的重合率 |
| mean_mask_agreement | `0.921656` | 每个 token keep/drop 决策的一致率 |

结论：当前轻量打分器可以约 `92%` 地复现 `topk_attention` 的 token 选择结果。

## 11. 推理时如何打分和剪枝

推理阶段加载 checkpoint：

```python
predictor = load_token_score_predictor(
    checkpoint_path=checkpoint_path,
    default_input_dim=1280,
    strict=True,
)
predictor.eval()
```

`predictor_early` 调用 vision tower：

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
    vispruner_min_tokens=1,
)
```

在 vision tower 第 8 层后：

```python
early_features = hidden_states.reshape(
    original_num_groups, self.spatial_merge_unit, -1
).mean(dim=1)[reverse_indices]

vispruner_scores = scorer(early_features).detach().float()

vispruner_keep_mask = self._vispruner_keep_mask_from_scores(
    vispruner_scores,
    image_lengths,
    vispruner_keep_ratio,
    vispruner_min_tokens,
)
```

然后裁剪 hidden states：

```python
keep_mask_window = vispruner_keep_mask[window_index]
group_hidden = hidden_states.reshape(
    original_num_groups, self.spatial_merge_unit, -1
)[keep_mask_window]
hidden_states = group_hidden.reshape(-1, group_hidden.shape[-1])
```

同步裁剪位置编码和 attention seqlens：

```python
position_embeddings = tuple(
    item.reshape(original_num_groups, self.spatial_merge_unit, -1)[keep_mask_window]
    .reshape(-1, item.shape[-1])
    for item in position_embeddings
)

cu_seqlens = self._vispruner_pruned_cu_seqlens(
    vispruner_keep_mask,
    image_lengths,
    grid_thw.device,
)
```

因此，后续 vision tower block 只会处理保留下来的 token。

## 12. 与第一版 topk_attention 的区别

| 方案 | 分数来源 | 是否需要完整 vision tower 后才能剪枝 | 是否减少 vision tower 后半段计算 |
|---|---|---|---|
| `topk_attention` | 完整 vision tower attention score | 是 | 否 |
| `predictor_score` | 完整 vision tower 后的 predictor score | 是 | 否 |
| `predictor_early` | 第 8 层 early hidden + MLP scorer | 否 | 是 |

当前真正带来高分辨率前半链路加速的是 `predictor_early`。

## 13. 高分辨率实验中的实际效果

最新 300 张 LeRobot 高分辨率图片实验：

```text
image_min_pixels = 254016
visual tokens: 324 -> 162
```

结果：

| 指标 | original | predictor_early | 变化 |
|---|---:|---:|---:|
| 视觉 token 数 | `324` | `162` | `-50.00%` |
| front_chain_total_time | `132.071 ms` | `98.886 ms` | `-33.186 ms (-25.13%)` |
| s02_vision_encode_or_prune | `98.171 ms` | `64.365 ms` | `-33.807 ms (-34.44%)` |

解释：

1. scorer 和剪枝本身会增加少量开销。
2. 但第 8 层后 token 数减半，vision tower 后半段计算下降。
3. 在 324-token 高分辨率场景中，节省明显大于新增开销。
4. 因此前半 token 处理链路总耗时下降约 `25.13%`。

## 14. 当前实现保留的开关

原始方案和第一版剪枝方案都没有删除，后续可通过配置切换：

| 配置 | 作用 |
|---|---|
| `vispruner_enable=False` | 原始 Wall-X，不剪枝。 |
| `strategy=topk_attention` | 第一版 attention teacher 剪枝。 |
| `strategy=predictor_score` | 完整 vision tower 后用 scorer 打分。 |
| `strategy=predictor_early` | 第 8 层后提前 scorer 打分并剪枝。 |
| `strategy=norm` | 使用 embedding norm 作为启发式分数。 |

这保证了后续实验可以在不同版本之间来回对比。

## 15. 关键结论

1. 当前打分器是一个 `1280 -> 320 -> 1` 的逐 token MLP。
2. 它的输入是 vision tower 第 8 层后的 `early_hidden` token feature。
3. 它的训练标签来自第一版 `topk_attention` teacher score。
4. 训练目标是回归标准化后的 teacher score，loss 是 MSE。
5. 30000 张 LeRobot 图片产生了 243 万个 token 级训练样本。
6. 当前 scorer 与 teacher 的 top-k token overlap 约为 `92.26%`。
7. 在高分辨率 324-token 场景中，`predictor_early` 将前半链路耗时从 `132.071 ms` 降到 `98.886 ms`。
