# 大创中期报告第5、6部分写作框架：Wall-X 视频输入与 ODE 推理优化

## 5. 调整 Wall-X 数据输入，以支持视频流

### 5.1 写作目标

这一节要说明：原始 Wall-X 主要按单帧图片输入做推理，本项目将输入链路扩展为真实视频 clip，使模型可以走 Qwen2.5-VL 原生的视频入口：

```text
MP4 / 视频帧序列
  -> 多帧 clip 解码
  -> <|video_pad|> prompt 占位
  -> processor 生成 pixel_values_videos
  -> video_grid_thw / second_per_grid_ts
  -> Wall-X 视觉编码器
  -> ODE action 推理
```

### 5.2 重点代码位置

- `wall_x/data/load_lerobot_dataset.py`
  - `media_type` 开关：第 88 行附近
  - `_video_preprocess()`：第 154 行附近
  - video 样本返回 `video_inputs`：第 192、220 行附近
  - collator 合并 `video_inputs`：第 476-505 行附近
  - 自动补 `second_per_grid_ts`：第 513-517 行附近
  - `delta_timestamps` video window：第 545 行附近

- `wall_x/data/utils.py`
  - processor 支持 `videos`、`pixel_values_videos`、`video_grid_thw`：第 167-198 行附近
  - `<|video_pad|>` token 数替换和越界检查：第 227-245 行附近
  - `get_wallx_normal_text(..., media_type="video")`：第 500-536 行附近

- `wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py`
  - `generate_flow_action()` 接收 `pixel_values_videos/video_grid_thw/second_per_grid_ts`：第 2812 行附近
  - 视频 embedding 编码与散射：第 2900-2920 行附近
  - video RoPE 使用 `second_per_grid_ts`：第 1661、1701 行附近

- `wall_x/serving/policy/utils.py`
  - serving 视频 schema：`obs["media_type"]="video"`、`obs["video_frames"][camera_key]=frames`：第 61-176 行附近
  - serving 注入 `second_per_grid_ts`：第 99-199 行附近

- `wall_x/serving/policy/wall_x_policy.py`
  - 声明支持 `image/video`：第 125-126 行附近

### 5.3 可画流程图

```text
数据集/服务输入
├── image 模式
│   ├── 单帧图片 [C,H,W]
│   ├── prompt 使用 <|image_pad|>
│   ├── processor 输出 pixel_values + image_grid_thw
│   └── 模型走 image vision path
│
└── video 模式
    ├── 多帧 clip [T,C,H,W]
    ├── prompt 使用 <|video_pad|>
    ├── processor 输出 pixel_values_videos + video_grid_thw
    ├── collator 补 second_per_grid_ts
    └── 模型走 video vision path
```

### 5.4 可引用实验结果

报告：`workspace/v2_inference_results/image_vs_video_inference_report.md`

- image action shape：`[1, 32, 20]`
- video action shape：`[1, 32, 20]`
- image tokens/features：`162 / 162`
- video tokens/features：`324 / 324`
- video_grid_thw：`[[2, 18, 18], [2, 18, 18]]`
- second_per_grid_ts：`[0.06666667, 0.06666667]`
- token/features 匹配：`True`

可写结论：视频路径已经不是“把视频当图片”，而是真正进入 `pixel_values_videos -> video_grid_thw -> video RoPE` 的视频输入链路。

---

## 6. Wall-X 的 ODE 推理部分优化

### 6.1 原始 ODE 推理逻辑

原始 Wall-X 的 action 不是一次性生成，而是先采样 noisy action，再通过 ODE/Flow Matching 多步修正。

```text
noisy_action
  -> prefetch t=0 更新一次
  -> postfix ODE step 1
  -> postfix ODE step 2
  -> ...
  -> postfix ODE step 9
  -> final action
```

重点代码：`wall_x/model/qwen2_5_based/modeling_qwen2_5_vl_act.py`

- `generate_flow_action()`：第 2790 行附近
- ODE 参数入口：第 2829-2836 行附近
- prefetch 首步更新：第 3047 行后
- postfix ODE 循环：第 3235 行后

---

## 6.2 版本一：V3 早停

### 思路

如果相邻 ODE step 的 action 变化很小，说明动作可能已经收敛，可以提前停止。

```text
if mean(abs(action_t - action_{t-1})) < threshold:
    stop early
```

### 重点代码

- 配置入口：`configuration_qwen2_5_vl.py` 第 233-237、306-310 行附近
- YAML 配置：`workspace/lerobot_example/config_qact_from_vlm.yml` 的 `ode_early_stop`
- 模型实现：`modeling_qwen2_5_vl_act.py` 第 3316-3408 行附近
- 测试脚本：
  - `scripts/profile_v3_ode_early_stop_lerobot_images.py`
  - `scripts/profile_v3_ode_early_stop_lerobot_videos.py`

### 视频数据结果

报告：`workspace/v3_early_stop/libero_50video_v3_ode_early_stop_report.md`

| case | total_ms | ODE_ms | actual_updates | action_MAE vs fixed |
|---|---:|---:|---:|---:|
| fixed_10 | 362.713 | 287.645 | 10 | 0.000000 |
| early_safe | 360.721 | 286.249 | 10 | 0.000000 |
| early_tradeoff | 296.564 | 222.011 | 8 | 0.519164 |

### 可写结论

早停方向可行，但阈值敏感。保守阈值几乎不停，激进阈值虽然减少耗时，但动作偏差较大。因此 V3 更适合作为探索版本，而不是最终稳定方案。

---

## 6.3 版本二：V5 复用中间结果 / ODE Velocity Cache

### 思路

相邻 ODE step 的 velocity 通常变化不大，因此不必每一步都重新计算 transformer。V5 在部分 step 直接复用上一次 velocity。

```text
fixed_10:
  step1 refresh
  step2 refresh
  step3 refresh
  ...

cache_i2:
  step1 refresh
  step2 refresh
  step3 reuse
  step4 refresh
  step5 reuse
  ...
```

### 重点代码

- 配置入口：`configuration_qwen2_5_vl.py` 第 238-240、311-313 行附近
- YAML 配置：`workspace/lerobot_example/config_qact_from_vlm.yml` 的 `ode_cache`
- 模型实现：`modeling_qwen2_5_vl_act.py`
  - `ode_cache_runtime`：第 3235 行附近
  - `step_with_kvcache()`：第 3243 行附近
  - cache hit 复用 velocity：第 3246-3258 行附近
  - refresh 正常计算 velocity：第 3261-3313 行附近
  - 输出 `ode_cache_info`：第 3420-3442 行附近
- 测试脚本：
  - `scripts/profile_v5_ode_cache_lerobot_images.py`
  - `scripts/profile_v5_ode_cache_lerobot_videos.py`

### 视频数据结果

报告：`workspace/v5_ode_cache/libero_50video_v5_ode_cache_report.md`

| case | total_ms | total 降幅 | ODE_ms | ODE 降幅 | cache_hit_rate | action_MAE vs fixed |
|---|---:|---:|---:|---:|---:|---:|
| fixed_10 | 369.372 | 0.00% | 291.709 | 0.00% | 0.00% | 0.000000 |
| cache_i2 | 238.762 | -35.36% | 162.192 | -44.40% | 44.44% | 0.008320 |
| cache_i3 | 176.839 | -52.12% | 99.170 | -66.00% | 66.67% | 0.013822 |
| early_tradeoff | 306.465 | -17.03% | 228.815 | -21.56% | 0.00% | 0.519293 |

### 可写结论

V5 相比早停更稳，因为它没有提前终止 ODE 轨迹，而是在固定步数框架内复用部分 velocity。`cache_i2` 是推荐配置，速度降低约 35%，MAE 只有约 0.0083；`cache_i3` 更快但误差略大。

---

## 6.4 版本三：V6 Student 学习，未来方向

### 思路

使用 fixed_10 作为 teacher，训练一个少步 student。student 不再靠手工 early stop 或 cache，而是学习如何用更少 ODE step 接近 teacher 的最终动作。

```text
teacher: 10 steps -> teacher action
student: 6 steps -> student action
loss = SmoothL1(student action, teacher action)
```

### 重点代码

- 蒸馏配置和 checkpoint 加载：`wall_x/model/ode_distill_utils.py`
  - `get_ode_distill_config()`：第 10 行附近
  - `apply_ode_distill_checkpoint()`：第 49 行附近
  - 默认禁止 V6 自动叠加 V5/V3：第 106-118 行附近
- 模型配置映射：`wall_x/model/model_utils.py` 第 168-229 行附近
- 训练脚本：`scripts/train_ode_distill_lerobot.py`
  - 数据根目录解析：第 45-54 行附近
  - V6 train_config：第 57-69 行附近
  - 禁止 V5/V3 runtime kwargs：第 130-133 行附近
  - student 训练与 eval：第 198 行后
- serving/infer 加载 student：`scripts/infer_robochallenge.py` 第 500-511、1071-1083 行附近
- V6 smoke 报告：`workspace/v6_ode_distill/smoke_step6/eval_report.md`
- V6 实现说明：`workspace/v6_ode_distill/v6_first_pass_implementation_report.md`

### 当前状态

已完成第一版 V6 代码链路 smoke：

- dataset input：`/root/autodl-tmp/wall_x/datasheet`
- resolved：`/root/autodl-tmp/wall_x/datasheet/libero_all`
- teacher：10 steps
- student：6 steps
- V5 cache：关闭
- V3 early stop：关闭

smoke 使用 `epochs=0`，只证明链路可跑通，不代表训练后精度。

### 可写未来方向

下一步正式训练 6-step student：

```bash
python3 scripts/train_ode_distill_lerobot.py   --dataset-root /root/autodl-tmp/wall_x/datasheet   --output-dir /root/autodl-tmp/wall_x/workspace/v6_ode_distill/ode_student_3000train_1000val_step6   --train-samples 3000   --val-samples 1000   --epochs 3   --student-num-inference-timesteps 6   --teacher-num-inference-timesteps 10   --device cuda
```

---

## 5、6部分整体汇报框架

```text
第5部分：视频输入支持
├── 原始问题：Wall-X 图片输入为主，视频数据不能完整走 video token 流
├── 修改目标：支持 image/video 双输入
├── 核心实现：media_type、video_inputs、<|video_pad|>、pixel_values_videos、video_grid_thw、second_per_grid_ts
├── 模型贯通：video RoPE + video vision encoder
└── 实验验证：video tokens/features 匹配，action shape 正确

第6部分：ODE 推理优化
├── 原始问题：fixed_10 ODE 多步推理耗时高
├── V3 早停
│   ├── 思路：动作变化小则提前停止
│   ├── 结果：激进早停有加速但误差大
│   └── 定位：探索方案
├── V5 复用
│   ├── 思路：复用相邻 step velocity
│   ├── 结果：cache_i2 稳定加速，MAE 很小
│   └── 定位：当前推荐方案
└── V6 Student
    ├── 思路：teacher-student 少步蒸馏
    ├── 当前：代码链路已跑通
    └── 定位：后续提升方向
```

## 推荐中期总结表达

本阶段完成了 Wall-X 在输入侧和推理侧的两类优化。输入侧将原本偏单帧图片的数据链路扩展为真实视频 clip 链路，使视频可以通过 `pixel_values_videos`、`video_grid_thw` 和 `second_per_grid_ts` 进入模型；推理侧围绕 ODE 多步动作生成进行了早停、velocity cache 和 student 蒸馏三类探索。实验表明，早停方案受阈值影响较大，激进配置会带来明显动作偏差；V5 velocity cache 在保持固定步数结构的同时复用中间 velocity，取得了更稳定的耗时下降和较低的动作误差，是当前阶段最具实用价值的优化。V6 student 蒸馏已完成第一版代码链路，为后续进一步减少 ODE 步数提供了方向。
