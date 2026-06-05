# Wall-X VisPruner 轻量打分器技术文档

本文档说明我们在 Wall-X 中训练轻量 token scorer 的完整流程，包括 teacher score 收集、MLP scorer 训练、评估指标、推理集成方式以及关键训练参数。

## 1. 研究目标

最初的 `topk_attention` 方案已经可以减少视觉 token，但它的分数来自完整 vision tower 的 attention，因此必须先跑完整视觉编码，再做 token 筛选。这样虽然主模型输入 token 变少了，但 vision tower 本身没有少算，甚至还会因为 `output_attentions=True` 增加开销。

因此我们训练一个轻量 scorer，目标是让它模仿 `topk_attention` 的 token 选择结果：

```text
当前 topk_attention 方案
图片 -> 完整 vision tower + attention score -> top-k 选 token -> 主模型

轻量 scorer / predictor_early 方案
图片 -> vision tower 前几层 early hidden -> 轻量 scorer 打分 -> 剪 token
     -> 后半段 vision tower 只处理保留 token -> 主模型
```

核心思路是：把 `topk_attention` 当成 teacher，把它产生的 `image_scores` 和 `keep_mask` 保存下来，再训练一个小 MLP 从早期视觉特征预测这些 teacher score。推理时，`predictor_early` 不再需要 `output_attentions=True`，而是在 vision tower 中间层提前打分并剪枝，从而减少 vision tower 后半段计算。

## 2. 整体技术流程

完整流程分为四步：

| 步骤 | 名称 | 作用 |
|---|---|---|
| Step 1 | 离线收集 teacher score | 用现有 `topk_attention` 方案跑 LeRobot 图片，保存每个视觉 token 的 teacher 分数和 keep mask。 |
| Step 2 | 训练轻量 scorer | 用 early hidden feature 作为输入，训练 MLP 去拟合 teacher score。 |
| Step 3 | 评估 scorer 是否像 teacher | 比较 scorer 选出的 top-k token 与 teacher keep mask 的重合率。 |
| Step 4 | 推理时 early pruning | 在 vision tower 第 8 层后调用 scorer，剪掉一半视觉 token，后续 vision tower 只处理保留 token。 |

这里的 teacher 不是人工标注，而是第一版 VisPruner `topk_attention` 自动生成的伪标签。它的意义是：先用一个准确但较贵的方法产生监督信号，再训练一个便宜的 predictor 去近似它。

## 3. Teacher Score 收集

### 3.1 代码位置

Teacher 收集脚本：

```text
scripts/collect_vispruner_teacher_scores.py
```

它加载 Wall-X 模型，并强制启用第一版 `topk_attention`：

```python
def build_train_config(model_path: str, keep_ratio: float) -> dict:
    return {
        "processor_path": model_path,
        "dof_config": model_cfg["dof_config"],
        "agent_pos_config": model_cfg["agent_pos_config"],
        "data": {
            "use_state_string_representation": False,
            "action_horizon_flow": 32,
        },
        "vispruner": {
            "enable": True,
            "strategy": "topk_attention",
            "keep_ratio": keep_ratio,
            "min_tokens": 1,
            "force_vision_eager": True,
        },
    }
```

关键调用如下：

```python
image_embeds, image_scores, predictor_features = model.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    output_attentions=True,
    return_vispruner_features=True,
    vispruner_feature_source=args.feature_source,
    vispruner_early_layer=args.early_layer,
)
```

这一步同时拿到三类信息：

| 字段 | 含义 |
|---|---|
| `image_embeds` | 完整 vision tower 输出后的视觉 embedding。 |
| `image_scores` | `topk_attention` teacher 产生的每个视觉 token 分数。 |
| `predictor_features` | 训练轻量 scorer 的输入特征，本项目最终使用 `early_hidden`。 |

然后用原来的 VisPruner top-k 逻辑生成 teacher keep mask：

```python
keep_mask, keep_indices = model.vispruner._build_image_keep_mask(
    image_embeds=image_embeds,
    image_scores=image_scores,
    image_lengths=image_lengths,
)
```

最终每个样本保存：

```python
record = {
    "dataset_index": int(dataset_idx),
    "image_scores": image_scores.detach().float().cpu(),
    "keep_mask": keep_mask.detach().cpu(),
    "keep_indices": [item.detach().cpu() for item in keep_indices],
    "predictor_features": cast_float_tensor(predictor_features, args.feature_dtype),
    "predictor_feature_source": args.feature_source,
    "predictor_early_layer": args.early_layer,
}
```

### 3.2 本项目实际收集参数

| 参数 | 实际值 |
|---|---|
| 数据集 | `/root/autodl-tmp/wall_x/datasheet/libero_all` |
| repo_id | `libero_all` |
| 图片字段 | `observation.images.faceImg` |
| 模型 | `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast` |
| teacher 策略 | `topk_attention` |
| keep_ratio | `0.5` |
| feature_source | `early_hidden` |
| early_layer | `8` |
| feature_dtype | `float16` |
| 收集图片数 | `30000` |
| 数据集总大小 | `271772` |
| shard 数 | `15` |
| 每个 shard 样本数 | `2000` |
| 保存目录 | `workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards` |

可复现命令如下：

```bash
python -u scripts/collect_vispruner_teacher_scores.py \
  --dataset-root /root/autodl-tmp/wall_x/datasheet/libero_all \
  --repo-id libero_all \
  --image-key observation.images.faceImg \
  --num-samples 30000 \
  --keep-ratio 0.5 \
  --feature-source early_hidden \
  --early-layer 8 \
  --feature-dtype float16 \
  --shard-size 2000 \
  --output-path /root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards
```

注意：当前 `token_score_predictor_30000_early_l8.pt` 的 teacher 数据来自 LeRobot 默认图像处理口径，平均每张图是 81 个视觉 token。因此 30000 张图片对应 `30000 * 81 = 2430000` 个 token 级训练样本。后续 324-token 高分辨率实验是推理阶段通过 `image_min_pixels=254016` 做的泛化测试。

## 4. 轻量 Scorer 模型结构

### 4.1 代码位置

模型定义：

```text
wall_x/model/vispruner_score_predictor.py
```

核心代码：

```python
class TokenScorePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or max(1, input_dim // 4))
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.net(token_features).squeeze(-1)
```

它是一个逐 token 打分的 MLP：

```text
early_hidden token feature -> Linear -> GELU -> Dropout -> Linear -> score
```

### 4.2 本项目实际模型参数

最终 checkpoint：

```text
workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt
```

checkpoint 中记录的模型结构：

| 参数 | 值 |
|---|---|
| input_dim | `1280` |
| hidden_dim | `320` |
| dropout | `0.0` |
| 输出维度 | `1` |
| 参数量 | 约 `410241` |

这个 scorer 很小，只负责给每个视觉 token 预测一个重要性分数。它不生成视觉 embedding，也不替代 vision tower；它只决定哪些 token 应该被保留。

## 5. Scorer 训练方法

### 5.1 代码位置

训练脚本：

```text
scripts/train_vispruner_score_predictor.py
```

### 5.2 输入与目标

训练时从 teacher record 中读取 feature 和 target：

```python
feature = record[resolved_feature_key].float()
```

最终使用的 feature 是：

```text
predictor_features = early_hidden features at vision layer 8
```

训练目标默认是 teacher score，而不是二值 mask：

```python
teacher_target = record["image_scores"].float().reshape(-1)
teacher_target = (teacher_target - teacher_target.mean()) / (
    teacher_target.std(unbiased=False) + 1e-6
)
```

也就是说，每张图内部的 teacher score 会先做标准化。这样做的原因是 top-k 剪枝关心的是 token 分数排序，而不是不同图片之间分数的绝对大小。

### 5.3 Loss

本项目最终使用 `target=score`，loss 是 MSE：

```python
def predictor_loss(pred, target, loss_type: str):
    if loss_type == "mask":
        return F.binary_cross_entropy_with_logits(pred.float(), target.float())
    return F.mse_loss(pred.float(), target.float())
```

训练目标可以理解为：

```text
让 scorer 预测出来的 token 排序尽量接近 topk_attention teacher 的 token 排序。
```

脚本也支持 `target=mask`，即直接学习二值 keep mask，并使用 BCE loss。但我们最终选的是 `target=score`，因为分数目标保留了 token 重要性的相对顺序信息，后续再通过 top-k 转成 keep mask。

### 5.4 本项目实际训练参数

| 参数 | 实际值 |
|---|---|
| teacher_path | `workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards` |
| output_path | `workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt` |
| feature_key | `auto`，实际解析为 `predictor_features` |
| target | `score` |
| device | `cuda:0` |
| epochs | `20` |
| batch_size | `4096` |
| learning_rate | `1e-3` |
| weight_decay | `1e-4` |
| dropout | `0.0` |
| seed | `42` |
| val_ratio | `0`，训练时使用全部 15 个 shard；训练后单独跑评估 |
| token 级训练样本数 | `2430000` |

可复现训练命令：

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

训练日志显示：

| 阶段 | loss |
|---|---:|
| epoch 1 train_loss | `0.157768` |
| epoch 10 train_loss | 约 `0.07` 量级 |
| epoch 19 train_loss | `0.062204` |
| epoch 20 train_loss | `0.062212` |
| checkpoint best_loss | `0.062204` |

## 6. Scorer 评估

评估脚本：

```text
scripts/evaluate_vispruner_score_predictor.py
```

评估时会用 predictor 给每个 token 打分，然后保留与 teacher 相同数量的 top-k token：

```python
pred_scores = predictor(features).detach().float().cpu().reshape(-1)
pred_mask = make_topk_mask(pred_scores, int(teacher_mask.sum().item()))
```

然后计算：

```python
overlap = (pred_mask & teacher_mask).sum().item() / teacher_mask.sum().item()
agreement = (pred_mask == teacher_mask).float().mean().item()
loss = F.mse_loss(pred_scores, teacher_scores).item()
```

本项目 30000 样本 scorer 的评估结果：

| 指标 | 数值 | 含义 |
|---|---:|---|
| records | `30000` | 评估图片数 |
| shards | `15` | teacher shard 数 |
| mean_score_mse | `0.076447` | predictor score 与 teacher score 的 MSE |
| mean_topk_overlap | `0.922611` | predictor 保留 token 与 teacher 保留 token 的重合率 |
| mean_mask_agreement | `0.921656` | 所有 token 位置上 keep/drop 决策一致率 |

可复现评估命令：

```bash
python -u scripts/evaluate_vispruner_score_predictor.py \
  --teacher-path /root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards \
  --checkpoint /root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt \
  --feature-key auto
```

## 7. 推理阶段如何调用 Scorer

### 7.1 保留原始方案与第一版剪枝方案

当前实现保留了多套策略，后续可以通过配置切换：

| 策略 | 含义 |
|---|---|
| `original` / `vispruner_enable=False` | 原始 Wall-X，不剪视觉 token。 |
| `topk_attention` | 第一版 VisPruner，用完整 vision tower attention score 选 token。 |
| `predictor_score` | 完整跑 vision tower，但用轻量 scorer 替代 attention score。 |
| `predictor_early` | 最新版本，在 vision tower 中间层提前打分并剪 token。 |
| `norm` | 用 embedding norm 作为简单启发式分数。 |

因此，原始方案和第一版剪枝方案没有被删除，后续仍可直接对比和回溯。

### 7.2 代码位置

推理入口：

```text
wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py
```

加载 predictor 的逻辑：

```python
def _build_vispruner_score_predictor(self):
    strategy = str(getattr(self.config, "vispruner_strategy", "original"))
    if strategy not in {"predictor_score", "predictor_early"}:
        return None

    checkpoint_path = getattr(self.config, "vispruner_predictor_path", None)
    predictor = load_token_score_predictor(
        checkpoint_path=checkpoint_path,
        default_input_dim=default_input_dim,
        strict=True,
    )
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)
    return predictor
```

`predictor_early` 的视觉编码调用：

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

### 7.3 Vision Tower 内部 early pruning

代码位置：

```text
wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py
```

关键逻辑发生在 vision tower 的 forward 内部：

```python
if (
    vispruner_early_prune
    and vispruner_feature_source == "early_hidden"
    and vispruner_keep_mask is None
    and layer_num + 1 >= early_layer
):
    early_features = hidden_states.reshape(
        original_num_groups, self.spatial_merge_unit, -1
    ).mean(dim=1)[reverse_indices]

    with torch.no_grad():
        vispruner_scores = scorer(early_features).detach().float()

    vispruner_keep_mask = self._vispruner_keep_mask_from_scores(
        vispruner_scores,
        image_lengths,
        vispruner_keep_ratio,
        vispruner_min_tokens,
    )
```

得到 keep mask 后，后续隐藏状态、位置编码和 attention seqlens 都同步裁剪：

```python
keep_mask_window = vispruner_keep_mask[window_index]
group_hidden = hidden_states.reshape(
    original_num_groups, self.spatial_merge_unit, -1
)[keep_mask_window]
hidden_states = group_hidden.reshape(-1, group_hidden.shape[-1])

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

这就是 `predictor_early` 能产生加速的关键：它不是完整 vision tower 后再删 token，而是在第 8 层后删 token，所以 vision tower 后半段只处理保留 token。

## 8. 当前 300 样本高分辨率实验验证

在最新高分辨率 LeRobot 300 图片实验中，我们使用：

```text
image_min_pixels = 254016
visual tokens: 324 -> 162
strategy: predictor_early
predictor_checkpoint: token_score_predictor_30000_early_l8.pt
```

前半部分 token 处理链路结果：

| 指标 | original | predictor_early | 变化 |
|---|---:|---:|---:|
| 视觉 token 数 | `324` | `162` | `-50.00%` |
| front_chain_total_time | `132.071 ms` | `98.886 ms` | `-33.186 ms (-25.13%)` |
| s02_vision_encode_or_prune | `98.171 ms` | `64.365 ms` | `-33.807 ms (-34.44%)` |
| s03_pruning_position_prepare | `0.000 ms` | `0.713 ms` | `+0.713 ms` |
| s04_apply_pruning | `0.000 ms` | `0.690 ms` | `+0.690 ms` |

这说明剪枝版确实增加了少量打分和裁剪相关开销，但 vision tower 主路径减少的时间更大，因此前半链路总耗时明显下降。

动作输出一致性结果：

| 指标 | 数值 |
|---|---:|
| MAE | `0.025359` |
| RMSE | `0.031705` |
| mean max_abs | `0.100451` |
| worst max_abs | `0.203187` |
| cosine_similarity | `0.999958` |
| allclose_rate, atol/rtol=1e-3 | `0.0000` |

解释：动作张量不是逐元素完全一致，但整体方向高度一致。后续若要判断是否真正可用于机器人部署，还需要接仿真 roll-out 或真实任务成功率，而不能只用逐元素 allclose。

## 9. 后续优化建议

1. 针对高分辨率 324-token 口径重新收集 teacher score 并重训 scorer。
   当前 scorer 训练数据是默认 81-token 口径，已经能在 324-token 上带来前半链路加速；但如果专门用 324-token teacher 训练，可能进一步提升 token 选择一致性和动作一致性。

2. 比较不同 early_layer。
   当前使用 `early_layer=8`。更早剪枝可能更快，但 token 判断可能更不准；更晚剪枝更稳，但节省的 vision tower 层数更少。建议测试 `early_layer=4/6/8/10`。

3. 加入任务成功率评估。
   scorer 的最终价值不是只让 token mask 接近 teacher，而是让机器人任务成功率不下降。因此后续应在仿真或真实机器人任务中评估成功率。

4. 为高分辨率场景单独建立报告口径。
   后续所有正式实验建议默认使用高分辨率大 token 设置，例如 `image_min_pixels=254016`，并统一报告 13 段前半链路时间戳。

## 10. 主要文件索引

| 文件 | 作用 |
|---|---|
| `scripts/collect_vispruner_teacher_scores.py` | 收集 `topk_attention` teacher score、keep mask、early hidden features。 |
| `scripts/train_vispruner_score_predictor.py` | 训练轻量 MLP scorer。 |
| `scripts/evaluate_vispruner_score_predictor.py` | 评估 scorer 与 teacher 的 top-k overlap / mask agreement。 |
| `wall_x/model/vispruner_score_predictor.py` | `TokenScorePredictor` 模型定义与 checkpoint 加载。 |
| `wall_x/model/vispruner_token_pruner.py` | 根据 score 构造 keep mask，并同步裁剪序列张量。 |
| `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl.py` | vision tower 内部 early pruning 实现。 |
| `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py` | Wall-X 动作模型中加载 scorer 与调用不同 VisPruner 策略。 |
| `workspace/vispruner_teacher_scores/libero_teacher_scores_all_early_l8_fp16_shards` | 30000 样本 teacher score shard 数据。 |
| `workspace/vispruner_teacher_scores/token_score_predictor_30000_early_l8.pt` | 当前最终使用的轻量 scorer checkpoint。 |
