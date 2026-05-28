# Wall-X 中 VisPruner 风格视觉 Token 筛选器接入报告

## 1. 总体思路

Wall-X 原始视觉输入流程为：

```text
图像 pixel_values
  -> Qwen2.5-VL vision tower
  -> image_embeds
  -> 替换 input_ids 中的 <image> token embedding
  -> language/action backbone
  -> action tokens
  -> ODE / flow action head
  -> robot action
```

加入 VisPruner 风格视觉筛选后，流程变为：

```text
图像 pixel_values
  -> vision tower 得到 image_embeds
  -> 给每个 image token 打分
  -> 按 keep_ratio 保留高分 token
  -> 同步裁剪 input_ids / attention_mask / position_ids / moe_token_types
  -> 将裁剪后的 image_embeds scatter 回文本序列
  -> 后续 Wall-X 主干照常推理
```

VisPruner 插入的位置是：

```text
视觉编码之后，进入语言/动作主干之前
```

它不修改 action head，也不修改 ODE，而是减少视觉 token 数量，从而降低后续 Transformer 处理的视觉上下文长度。

## 2. 配置入口

配置定义在：

```text
wall_x/model/qwen2_5_based/configuration_qwen2_5_vl.py
```

核心字段如下：

```python
vispruner_enable=False,
vispruner_strategy="original",
vispruner_keep_ratio=1.0,
vispruner_min_tokens=1,
vispruner_force_vision_eager=True,
vispruner_predictor_path=None,
vispruner_predictor_source="image_embeds",
```

含义：

- `vispruner_enable`：是否启用视觉 token 筛选。
- `vispruner_strategy`：筛选策略，如 `topk_attention`、`norm`、`predictor_score`。
- `vispruner_keep_ratio`：保留比例，例如 `0.5` 表示保留 50% 视觉 token。
- `vispruner_min_tokens`：每张图至少保留多少 token。
- `vispruner_force_vision_eager`：使用 attention 打分时，强制 vision attention 走 eager，方便拿到 attention 权重。

训练配置会通过：

```text
wall_x/model/model_utils.py
```

写入模型配置：

```python
def apply_vispruner_config(train_config, model_config):
    vispruner_config = train_config.get("vispruner", {}) or {}

    model_config.vispruner_enable = bool(vispruner_config.get("enable", ...))
    model_config.vispruner_strategy = str(vispruner_config.get("strategy", ...))
    model_config.vispruner_keep_ratio = float(vispruner_config.get("keep_ratio", ...))
    model_config.vispruner_min_tokens = int(vispruner_config.get("min_tokens", ...))
```

## 3. 模型初始化中插入 Pruner

核心文件：

```text
wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py
```

初始化时，如果使用 `topk_attention` 策略，需要拿到视觉 attention 权重，因此强制 vision tower 使用 eager attention：

```python
if (
    getattr(config, "vispruner_enable", False)
    and getattr(config, "vispruner_strategy", "original") == "topk_attention"
    and getattr(config, "vispruner_force_vision_eager", True)
):
    config.vision_config._attn_implementation = "eager"
```

随后创建筛选器：

```python
self.vispruner = WallXVisPruner(config)
self.vispruner_score_predictor = None
```

筛选器实现文件：

```text
wall_x/model/vispruner_token_pruner.py
```

核心类：

```python
class WallXVisPruner(nn.Module):
    SCORE_BASED_STRATEGIES = {
        "topk_attention",
        "predictor_score",
        "predictor_early",
        "norm",
    }
```

## 4. 是否启用剪枝的判断

模型中通过 `_should_prune_images()` 判断当前输入是否需要剪枝：

```python
def _should_prune_images(self, pixel_values, image_grid_thw) -> bool:
    return (
        pixel_values is not None
        and image_grid_thw is not None
        and self.vispruner.enabled
    )
```

`self.vispruner.enabled` 的逻辑为：

```python
@property
def enabled(self) -> bool:
    return (
        bool(getattr(self.config, "vispruner_enable", False))
        and self.strategy != "original"
        and self.keep_ratio < 1.0
    )
```

因此必须同时满足：

```text
vispruner_enable=True
strategy != "original"
keep_ratio < 1.0
```

才会真正进行视觉 token 筛选。

## 5. 图像编码阶段插入剪枝

核心入口函数：

```python
def _encode_images_and_maybe_prune(...):
    pixel_values = pixel_values.type(self.visual.dtype)

    if self._should_prune_images(pixel_values, image_grid_thw):
        strategy = str(getattr(self.config, "vispruner_strategy", "original"))
```

如果策略是 `topk_attention`，让 vision tower 返回视觉 token embedding 和 attention score：

```python
image_embeds, image_scores = self.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    output_attentions=True
)
```

然后调用 VisPruner：

```python
prune_result = self.vispruner(
    image_embeds=image_embeds,
    image_scores=image_scores,
    input_ids=input_ids,
    image_grid_thw=image_grid_thw,
    image_token_id=self.config.image_token_id,
    spatial_merge_size=self.config.vision_config.spatial_merge_size,
    attention_mask=attention_mask,
    labels=labels,
    moe_token_types=moe_token_types,
    position_ids=position_ids,
    pad_token_id=self._pad_token_id(),
)
```

返回的是裁剪后的：

```text
image_embeds
input_ids
attention_mask
labels
moe_token_types
position_ids
```

## 6. Token 打分与 Top-K 保留

在 `WallXVisPruner` 中，先根据策略得到每个视觉 token 的分数：

```python
if self.strategy == "norm":
    scores = image_embeds.detach().float().norm(dim=-1)
elif image_scores is None:
    scores = image_embeds.detach().float().norm(dim=-1)
else:
    scores = image_scores.detach().float().reshape(-1).to(image_embeds.device)
```

然后每张图内部按 `keep_ratio` 保留 top-k：

```python
keep_count = int(torch.ceil(torch.tensor(length * self.keep_ratio)).item())
keep_count = max(self.min_tokens, keep_count)
keep_count = min(length, keep_count)

local_scores = scores[offset : offset + length]
local_keep = torch.topk(local_scores, k=keep_count, largest=True).indices
local_keep = local_keep.sort().values
keep_mask[offset + local_keep] = True
```

关键点：

```text
不是全 batch 混在一起选，而是每张图单独选 top-k。
```

这样可以避免某一张图的 token 被另一张图的高分 token 挤掉。

## 7. 同步裁剪文本序列中的 image token

只裁剪 `image_embeds` 不够，因为 `input_ids` 里也有对应数量的 `<image>` 占位 token。两者必须严格对齐。

先找到当前样本中的 image token 位置：

```python
image_positions = (
    (input_ids[batch_idx] == image_token_id)
    .nonzero(as_tuple=False)
    .flatten()
)
```

构造序列保留 mask：

```python
seq_keep = torch.ones(
    input_ids.shape[1],
    dtype=torch.bool,
    device=input_ids.device
)

seq_keep[image_positions] = torch.cat(sample_image_keep).to(input_ids.device)
```

然后同步裁剪：

```python
new_input_ids.append(input_ids[batch_idx, seq_keep])

if attention_mask is not None:
    new_attention_mask.append(attention_mask[batch_idx, seq_keep])

if moe_token_types is not None:
    new_moe_token_types.append(moe_token_types[batch_idx, seq_keep])

if position_ids is not None:
    new_position_ids.append(position_ids[:, batch_idx, seq_keep])
```

这一步说明 VisPruner 不是只丢掉视觉 embedding，而是把整条 multimodal token 序列中对应的视觉 token 也删除。

## 8. 重新 Padding 与位置编码修正

不同样本剪枝后长度可能不同，因此需要重新 padding：

```python
pruned_input_ids = self._pad_1d(new_input_ids, pad_token_id)
pruned_attention_mask = self._pad_1d(new_attention_mask, 0)
pruned_moe_token_types = self._pad_1d(new_moe_token_types, 0)
pruned_position_ids = self._pad_position_ids(new_position_ids)
```

然后重新计算 RoPE delta：

```python
rope_deltas = self._compute_rope_deltas(
    pruned_position_ids,
    pruned_attention_mask
)
```

这一步用于保证 Qwen2.5-VL 的位置编码在视觉 token 数量变化后仍然一致。

## 9. 裁剪后重新进入 Wall-X 主干

模型调用：

```python
image_embeds, input_ids, attention_mask, labels, moe_token_types, position_ids, image_pruned = \
    self._encode_images_and_maybe_prune(...)
```

然后重新生成 token embedding：

```python
inputs_embeds = self.model.embed_tokens(input_ids)
```

再把裁剪后的 `image_embeds` scatter 回 `<image>` token 对应位置：

```python
inputs_embeds = self._scatter_image_embeds(
    inputs_embeds,
    input_ids,
    image_embeds
)
```

`_scatter_image_embeds()` 会检查两者数量是否一致：

```python
n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
n_image_features = image_embeds.shape[0]

if n_image_tokens != n_image_features:
    raise ValueError(...)
```

这保证剪枝后的 image token 数量和 image embedding 数量严格匹配。

## 10. 对 Wall-X 推理链路的影响

剪枝发生在视觉 token 进入大模型主干之前，因此后续流程仍然是原来的 Wall-X：

```text
裁剪后的 multimodal inputs_embeds
  -> Qwen2.5-VL / MoE backbone
  -> action token hidden states
  -> action_preprocessor.action_proj_back
  -> flow / ODE action generation
  -> predict_action
```

优点：

```text
不需要改 action head
不需要改 ODE
不需要改机器人动作格式
只减少视觉 token 数
```

但如果 Wall-X 的主要耗时集中在 ODE/action head 阶段，那么只减少视觉 token，端到端时间下降可能不明显。

## 11. 当前实验结果说明

在已有实验中，图片样本视觉 token 从：

```text
81 -> 41
```

约减少 50%。

但端到端推理时间下降不明显，原因是：

```text
视觉 token 减少主要影响 embed / prefill / transformer 序列长度；
但 Wall-X 的主要耗时集中在 ODE 反复调用 action transformer；
ODE 阶段仍然要多次处理 action/postfix token。
```

因此，VisPruner 插入是成功的，视觉 token 也确实减少了；只是 Wall-X 当前瓶颈不完全在视觉 token 数量上。

## 12. 总结

本项目将 VisPruner 风格筛选器插入到 Wall-X 的图像编码后、主干 Transformer 前：

```text
image_embeds 生成后
  -> 对 image token 打分
  -> 按 keep_ratio 保留高分 token
  -> 同步裁剪 input_ids / attention_mask / position_ids / moe_token_types
  -> 将裁剪后的 image_embeds scatter 回序列
  -> 保持后续 Wall-X action 推理流程不变
```

这种接入方式的特点是模块化、低侵入：它不改变动作预测头、不改变 ODE 逻辑、不改变机器人动作输出格式，只在视觉 token 进入主干前做筛选。
