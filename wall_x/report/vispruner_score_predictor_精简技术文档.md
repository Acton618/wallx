# VisPruner 轻量打分器精简技术文档

## 1. 为什么要训练轻量打分器

第一版 `topk_attention` 剪枝已经能把视觉 token 减少一半，但它的 token 分数来自完整 vision tower 的 attention：

```text
图片 -> 完整 vision tower + output_attentions=True -> attention score -> top-k 剪枝
```

问题是：必须先完整跑完 vision tower，才能拿到 attention score。因此它主要减少主模型输入 token，对 vision tower 本身的计算节省有限。

我们训练轻量 scorer 的目的就是提前预测 token 重要性：

```text
图片 -> vision tower 前 8 层 early hidden -> 轻量 scorer 打分 -> 剪掉一半 token
     -> 后半段 vision tower 只处理保留 token -> 主模型
```

这样不仅减少主模型输入 token，也减少 vision tower 后半段计算。

## 2. 整体训练流程

| 步骤 | 做什么 | 产物 |
|---|---|---|
| Step 1 | 用第一版 `topk_attention` 当 teacher，离线跑官方 LeRobot 图片 | 每个 token 的 `image_scores` 和 `keep_mask` |
| Step 2 | 保存 vision tower 第 8 层的 `early_hidden` 特征 | scorer 的输入特征 `predictor_features` |
| Step 3 | 训练小 MLP，让它预测 teacher score | `token_score_predictor_30000_early_l8.pt` |
| Step 4 | 推理时使用 `predictor_early`，在第 8 层后提前剪 token | 高分辨率 324 -> 162 token，前半链路加速 |

一句话概括：  
先用准确但较慢的 `topk_attention` 生成伪标签，再训练一个很小的 MLP 去模仿它，最后用这个 MLP 在 vision tower 中间层提前剪枝。

## 3. Teacher Score 如何收集

脚本：

```text
scripts/collect_vispruner_teacher_scores.py
```

关键逻辑：

```python
image_embeds, image_scores, predictor_features = model.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    output_attentions=True,
    return_vispruner_features=True,
    vispruner_feature_source="early_hidden",
    vispruner_early_layer=8,
)

keep_mask, keep_indices = model.vispruner._build_image_keep_mask(
    image_embeds=image_embeds,
    image_scores=image_scores,
    image_lengths=image_lengths,
)
```

保存内容：

```python
record = {
    "image_scores": image_scores,
    "keep_mask": keep_mask,
    "keep_indices": keep_indices,
    "predictor_features": predictor_features,
    "predictor_feature_source": "early_hidden",
    "predictor_early_layer": 8,
}
```

这里 `image_scores` 和 `keep_mask` 是 teacher 标签，`predictor_features` 是轻量 scorer 的训练输入。

## 4. 轻量 Scorer 结构

代码：

```text
wall_x/model/vispruner_score_predictor.py
```

模型是一个逐 token 打分的小 MLP：

```python
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

本项目实际结构：

```text
early_hidden token feature: 1280 维
MLP hidden_dim: 320
输出: 每个 token 一个 score
```

它不替代 vision tower，只负责回答一个问题：这个视觉 token 值不值得保留？

## 5. 训练目标与 Loss

训练脚本：

```text
scripts/train_vispruner_score_predictor.py
```

我们训练的是 teacher score，而不是直接训练二值 mask：

```python
teacher_target = record["image_scores"].float().reshape(-1)
teacher_target = (teacher_target - teacher_target.mean()) / (
    teacher_target.std(unbiased=False) + 1e-6
)
loss = F.mse_loss(pred.float(), teacher_target.float())
```

原因：剪枝本质上依赖 token 排序。学习连续 score 比直接学习 0/1 mask 更能保留 token 重要性的相对顺序。

## 6. 关键训练参数

| 参数 | 数值 |
|---|---|
| 数据集 | `/root/autodl-tmp/wall_x/datasheet/libero_all` |
| repo_id | `libero_all` |
| 图片字段 | `observation.images.faceImg` |
| 模型 | `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast` |
| teacher 策略 | `topk_attention` |
| keep_ratio | `0.5` |
| feature_source | `early_hidden` |
| early_layer | `8` |
| 收集图片数 | `30000` |
| shard 数 | `15` |
| 每个 shard | `2000` 张图片 |
| feature_dtype | `float16` |
| 默认 token 数 | `81` / 图 |
| token 级训练样本数 | `2430000` |
| scorer input_dim | `1280` |
| scorer hidden_dim | `320` |
| dropout | `0.0` |
| epochs | `20` |
| batch_size | `4096` |
| learning_rate | `1e-3` |
| weight_decay | `1e-4` |
| seed | `42` |
| checkpoint | `workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt` |

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

## 7. 训练与评估结果

训练 loss：

| 阶段 | loss |
|---|---:|
| epoch 1 | `0.157768` |
| epoch 20 | `0.062212` |
| best_loss | `0.062204` |

评估结果：

| 指标 | 数值 | 含义 |
|---|---:|---|
| mean_score_mse | `0.076447` | scorer score 与 teacher score 的均方误差 |
| mean_topk_overlap | `0.922611` | scorer 选中的 token 与 teacher 选中 token 的重合率 |
| mean_mask_agreement | `0.921656` | 每个 token 保留/丢弃决策的一致率 |

结论：轻量 scorer 能约 92% 地复现第一版 `topk_attention` 的 token 选择结果。

## 8. 推理时如何使用

最新推理策略是：

```text
strategy = predictor_early
predictor_source = early_hidden
predictor_early_layer = 8
keep_ratio = 0.5
```

关键代码位置：

```text
wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py
wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py
```

核心调用：

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
early_features = hidden_states.reshape(...).mean(dim=1)
vispruner_scores = scorer(early_features)
vispruner_keep_mask = topk(vispruner_scores, keep_ratio=0.5)
hidden_states = hidden_states[vispruner_keep_mask]
```

剪枝后，后续 vision tower block 只处理保留下来的 token。

## 9. 最新高分辨率 300 样本实验结果

实验设置：

```text
LeRobot 300 张图片
image_min_pixels = 254016
original vs predictor_early
```

核心结果：

| 指标 | original | predictor_early | 变化 |
|---|---:|---:|---:|
| 视觉 token | `324` | `162` | `-50.00%` |
| front_chain_total_time | `132.071 ms` | `98.886 ms` | `-33.186 ms (-25.13%)` |
| vision encode / prune 主路径 | `98.171 ms` | `64.365 ms` | `-33.807 ms (-34.44%)` |

这说明在高分辨率大 token 场景下，`predictor_early` 的优势是明确的：虽然新增了少量 scorer 与剪枝开销，但 vision tower 后半段少算的部分更大，最终前半 token 处理链路明显加速。

## 10. 需要强调的结论

1. `topk_attention` 是 teacher，不是最终加速方案。
2. 轻量 scorer 学的是 teacher 的 token 重要性排序。
3. 当前 scorer 使用 30000 张 LeRobot 图片训练，token 级样本数是 243 万。
4. 最新 `predictor_early` 没有删除原始方案和第一版剪枝方案，后续仍可通过开关对比。
5. 对高分辨率 324-token 场景，前半 token 处理链路已经从 `132.071 ms` 降到 `98.886 ms`。
6. 后续最值得继续做的是：用 324-token 高分辨率 teacher 重新收集数据并重训 scorer，再比较 token 选择一致性和动作输出一致性。
