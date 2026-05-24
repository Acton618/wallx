wall_x 时间节点与复杂度评估交接报告

面向机器人的轻量化 VLA 算法与高效部署研究

生成日期：2026 年 5 月 23 日。对象：vispruner 移植后的 wall_x 视觉语言动作预测模型。文档目的：说明为什么要统计时间节点和复杂度、当前采用了什么方案、如何运行验证、已经得到什么结果，以及这些结果应该如何解读。

一、先说明我们在做什么

本项目研究的是面向机器人的 VLA（Vision-Language-Action）模型。简单理解，模型输入图像、语言指令和机器人状态，输出机器人下一段动作。vispruner 的作用是减少进入后续 Transformer 的视觉 token 数量，也就是把图像信息压缩成更少的视觉 token，从而希望减少计算量并提升推理速度。

因此当前任务不是重新训练模型，也不是改变模型输出逻辑，而是在现有推理流程中增加两类观测指标：第一类是实际推理时间，用来回答“模型真的跑得更快了吗”；第二类是理论复杂度，用来回答“视觉 token 减少后，理论计算量是否下降，以及下降多少”。

这两类指标需要同时存在。时间统计来自真实程序运行，受 GPU、显存、算子实现和 ODE integration 等因素影响；复杂度统计来自公式估算，重点反映剪枝前后 token 数变化带来的理论收益。

二、为什么不能只看运行时间

只看运行时间会有一个问题：GPU 推理速度受到很多工程因素影响，例如 CUDA kernel 调度、显存带宽、cache、batch 大小、自定义算子、ODE 多步循环等。即使视觉 token 明显减少，最终耗时也未必按同样比例下降。

复杂度统计的价值在于把“理论计算代价”单独拿出来看。对于 Transformer 的 attention 部分，序列长度越长，attention 矩阵越大，计算量通常近似随序列长度平方增长。因此，当视觉 token 被剪掉后，总序列长度变短，attention 理论复杂度会明显下降。

三、采用的核心公式

本次复杂度统计不是使用 fvcore 对完整 generate_flow_action / diffusion 链路做 tracing，而是基于 token 数量做稳定估算。原因是当前完整动作预测路径包含视觉编码、embedding 拼接、MoE token 分组、KV-cache、prefix/postfix 拆分、自定义 fusions ops、ODE integration 多步循环和 action head，动态控制流较多，fvcore 对这类路径不稳定。

当前使用的核心公式如下。这里的 seq_len 表示进入 Transformer 的总序列长度，由文本 token、视觉 token 和 action token 共同组成。

这些公式的含义是：先分别计算剪枝前和剪枝后的总序列长度，再根据 attention 的平方复杂度估算两者比例，最后得到理论 attention 计算量下降百分比。

四、代码中做了什么

| 文件 | 位置或开关 | 作用说明 |
| --- | --- | --- |
| model/qwen2_5_based/modeling_qwen2_5_vl_act.py | _InferencePerfTimer | 使用 CUDA Event 记录 GPU 推理耗时，并在 CPU 场景下保留 perf_counter 兜底。该计时默认关闭，只有显式打开 profile_timing 才会输出。 |
| model/qwen2_5_based/modeling_qwen2_5_vl_act.py | _log_complexity_track(...) | 从 input_ids、attention_mask 和 token id 中统计 vision_tokens_before、text_tokens、action_tokens，并根据 vision_keep_ratio 或 vision_tokens_after 得到剪枝后的视觉 token 数。 |
| model/qwen2_5_based/modeling_qwen2_5_vl_act.py | generate_flow_action(..., profile_timing, profile_complexity) | 在动作预测路径中挂接时间统计和复杂度统计，但不改变模型输出 tensor。 |
| model/model_utils.py | num_floating_point_operations(...) | 作为辅助函数复用，用于估算剪枝前后粗粒度 FLOPs 变化；该值是理论估算，不是硬件真实 profiler 输出。 |
| serving/policy/wall_x_policy.py | WALLX_PROFILE_TIMING | 服务端推理时可通过环境变量打开时间统计，默认关闭，不影响正常部署。 |

需要强调的是：这些改动的定位是“观测和统计”，不是“修改模型行为”。默认情况下不打开 profiling，模型正常推理不会每次都强制统计。只有测试时设置 profile_timing、profile_complexity 或传入 vision_keep_ratio 等参数，才会打印对应日志。

五、运行方式

本次实验是在 wallx 环境中执行的，命令如下。命令中的 warmup=5 表示前 5 次推理只用于预热 GPU，不纳入最终平均耗时；iters=20 表示正式统计 20 次推理平均时间；vision_keep_ratio=0.5 表示模拟保留约 50% 的视觉 token。

source /root/miniconda3/etc/profile.d/conda.sh

conda activate wallx

export PYTHONPATH=/root/autodl-tmp/wall_x:$PYTHONPATH

python /tmp/wallx_profile_once.py --warmup 5 --iters 20 --vision_keep_ratio 0.5

六、复杂度输出结果

| 指标 | 数值 | 解释 |
| --- | --- | --- |
| vision_tokens_before | 81 | 剪枝前，从输入序列中统计到的视觉 token 数。 |
| vision_tokens_after | 41 | 剪枝后估算的视觉 token 数。本次使用 vision_keep_ratio=0.5，因此 81 个视觉 token 约保留为 41 个。 |
| text_tokens | 53 | 文本指令、系统提示等非视觉、非 action 的 token 数。 |
| action_tokens | 32 | 动作预测相关 token 数，对应 action horizon 中用于动作输出的 token。 |
| seq_len_before | 166 | 剪枝前总序列长度，计算为 53 + 81 + 32。 |
| seq_len_after | 126 | 剪枝后总序列长度，计算为 53 + 41 + 32。 |
| attention_ratio | 0.576136 | 剪枝后 attention 理论复杂度与剪枝前的比例。 |
| estimated_attention_reduction | 42.39% | 按 attention 平方复杂度估算得到的理论 attention 计算量下降比例。 |
| estimated_train_flops_before | 1,509,725,809,920 | 复用 num_floating_point_operations(...) 得到的剪枝前粗粒度训练级 FLOPs 估算。 |
| estimated_train_flops_after | 1,357,458,597,120 | 复用 num_floating_point_operations(...) 得到的剪枝后粗粒度训练级 FLOPs 估算。 |
| estimated_train_flops_ratio | 0.899142 | 剪枝后粗粒度 FLOPs 与剪枝前的比例。 |
| estimated_train_flops_reduction | 10.09% | 粗粒度 FLOPs 估算下降比例。 |

将本次实际 token 数代入公式，可以得到下面的推导过程。

七、时间统计输出结果

时间统计使用 CUDA Event 完成，因此它反映的是本次程序在 GPU 上实际跑出来的耗时。下面表格中的“最后一次耗时”来自最后一轮推理日志，“平均耗时”来自 20 次正式推理的平均 total_time。

| 计时项 | 最后一次耗时 | 说明 |
| --- | --- | --- |
| vision_image_forward | 26.506 ms | 视觉塔处理图像的耗时。 |
| embed_processing | 26.840 ms | embedding 和多模态输入拼接处理耗时；该项包含视觉特征写入输入 embedding 的过程。 |
| position_encoding | 0.573 ms | RoPE / position id 等位置编码准备耗时。 |
| action_initialization | 0.363 ms | diffusion / flow action 初始噪声与 action embedding 初始化耗时。 |
| prefetch_forward | 29.787 ms | prefix 阶段 Transformer forward 耗时。 |
| cache_preprocessing | 1.150 ms | KV-cache 裁剪和整理耗时。 |
| ode_integration | 276.320 ms | ODE integration 多步动作生成耗时，是当前完整推理链路中的主要耗时部分。 |
| postprocessing | 0.010 ms | 动作反归一化与输出组装耗时。 |
| total_time | 335.468 ms | 最后一次完整动作预测推理耗时。 |
| Average total_time over 20 runs | 339.891 ms | 20 次正式统计的平均完整推理耗时，报告中建议优先引用该值。 |

需要注意，时间分项中存在嵌套关系，例如 vision_image_forward 属于 embed_processing 内部的一部分，因此不要把所有分项简单相加当作 total_time。报告中最稳妥的速度指标是 Average total_time over 20 runs = 339.891 ms。

八、如何解读这次结果

从复杂度角度看，视觉 token 从 81 减少到 41，总序列长度从 166 降到 126。由于 attention 的主要计算近似与序列长度平方相关，因此理论 attention 复杂度下降到原来的 57.61%，对应下降 42.39%。这说明视觉 token 剪枝确实会显著降低 Transformer attention 部分的理论计算压力。

从粗粒度 FLOPs 角度看，estimated_train_flops 从约 1.510 × 10^12 降到 1.357 × 10^12，下降 10.09%。这个下降幅度小于 attention 复杂度下降幅度，是正常现象，因为全模型 FLOPs 不只包含 attention，还包括视觉编码、MLP、投影层、action head 等其他计算。

从实际时间角度看，完整动作预测平均耗时为 339.891 ms。当前耗时最大的部分是 ODE integration，最后一次为 276.320 ms。这意味着即使 attention 复杂度下降明显，完整推理耗时也不会按 42.39% 等比例下降，因为完整链路中还有大量与视觉 token 数无关或弱相关的计算。

九、为什么不用 fvcore 直接测完整 FLOPs

之前尝试过使用 fvcore.nn.FlopCountAnalysis 对完整 forward / action prediction 路径做 tracing，但在当前模型上出现过 NoneType object has no attribute layers 等错误。这不是简单的语法错误，而是因为完整动作预测路径里包含 KV-cache、动态控制流、自定义 fusions ops、MoE token 分组和 ODE 多步循环。fvcore 更适合标准、静态、单次 forward 的 PyTorch module，对这类复杂推理链路支持不好。

因此当前方案选择基于 token 数进行理论复杂度估算。它的优点是稳定、可解释、不会引入额外依赖，也不会影响模型输出；局限是它不是硬件 profiler，也不是全模型逐算子的精确 FLOPs 统计。

十、结论

| 问题 | 本次结论 |
| --- | --- |
| 有没有拿到剪枝前后 token 数？ | 已拿到。vision_tokens_before=81，vision_tokens_after=41，text_tokens=53，action_tokens=32。 |
| 理论 attention 复杂度是否下降？ | 已下降。seq_len 从 166 降到 126，attention 理论复杂度下降 42.39%。 |
| 粗粒度 FLOPs 是否下降？ | 已下降。estimated_train_flops_reduction=10.09%。 |
| 实际推理耗时是多少？ | 20 次平均 total_time 为 339.891 ms，最后一次 total_time 为 335.468 ms。 |
| 这是不是精准硬件 FLOPs？ | 不是。复杂度部分是基于 token 数和模型配置的理论估算；时间部分是 CUDA Event 真实测量。 |
| 是否影响模型正常运行？ | 不影响。统计逻辑默认关闭，只有显式打开 profiling 或传入复杂度参数时才会输出日志。 |

十一、后续建议

如果后续 vispruner 真正接入推理路径，建议把实际剪枝后的视觉 token mask 或 vision_tokens_after 直接传入 _log_complexity_track，而不是继续用 vision_keep_ratio 模拟。这样报告中的 vision_tokens_after 会来自真实剪枝结果。

如果后续需要更细粒度的性能分析，可以分模块统计 visual encoder FLOPs、prefill / prefix Transformer FLOPs、postfix Transformer FLOPs 和 action head FLOPs。但这属于下一阶段的细化 profiling，不建议再强行对完整 diffusion 链路做一次性 fvcore tracing。

GitHub 仓库：git@github.com:Acton618/wallx.git。本文档文件：wall_x/wallx_time_flops_handoff_report.docx。
