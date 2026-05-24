import html
import json
from pathlib import Path


RESULTS_JSON = Path(
    "/root/autodl-tmp/wall_x/wall_x/report/wallx_vispruner_30image_timing_results.json"
)
REPORT_DOC = Path(
    "/root/autodl-tmp/wall_x/wall_x/report/wallx_vispruner_30image_timing_report_cn.doc"
)


SEGMENT_DESCRIPTIONS = {
    "external_prepare_batch_ms": (
        "模型外部输入准备",
        "从原始图片、prompt、机器人状态构造模型 batch 的耗时，包括图像读取后的 processor 处理、tokenizer、状态归一化和张量搬运等。它不属于模型 forward 内部，但属于真实服务端端到端链路。",
        "该段基本不受 VisPruner 影响；如果这里差异很大，通常是数据预处理波动，而不是剪枝算法收益。",
    ),
    "total_time": (
        "模型内部完整动作生成",
        "进入 generate_flow_action 后到输出动作张量前的总耗时，覆盖视觉编码/剪枝、embedding 拼接、位置编码、prefill、KV-cache 处理、ODE 多步动作生成和后处理。",
        "这是判断完整 Wall-X 动作推理是否真正变快的核心指标。",
    ),
    "embed_processing": (
        "多模态 embedding 准备总段",
        "包含图像路径、token embedding、图像特征 scatter、视频特征处理、proprioception embedding 和 attention_mask 设备对齐。",
        "该段内含 image_path_total，因此不能和 image_path_total 相加。它能反映剪枝前的视觉打分和序列裁剪是否增加了前处理成本。",
    ),
    "image_path_total": (
        "图像路径总段",
        "从 pixel_values dtype 转换开始，到图像视觉特征可用于 scatter 结束。baseline 是普通 vision encode；pruned 是 vision encode with output_attentions + position ids 准备 + VisPruner 裁剪。",
        "如果 pruned 没有降低该段，说明当前剪枝发生在 vision tower 之后，不能节省视觉编码本身，还会增加打分/裁剪成本。",
    ),
    "vision_image_forward": (
        "兼容旧命名的图像路径总段",
        "包住 _encode_images_and_maybe_prune 的整体耗时，语义与 image_path_total 接近，用于兼容旧报告。",
        "用于和旧 profiling 报告对齐；分析时优先看 image_path_total 及其子段。",
    ),
    "image_cast": (
        "图像 dtype 转换",
        "将 pixel_values 转为 visual tower 使用的 dtype。",
        "通常很小，不是瓶颈。",
    ),
    "vision_image_encode": (
        "baseline 普通视觉编码",
        "关闭剪枝时调用 self.visual(pixel_values, grid_thw=image_grid_thw) 的耗时。",
        "这是 baseline 图像塔成本。pruned 路径不会出现该 key，而是使用 vision_image_encode_score。",
    ),
    "vision_image_encode_score": (
        "pruned 视觉编码与 attention score 获取",
        "开启剪枝时调用 self.visual(..., output_attentions=True) 的耗时，用于得到 image_embeds 和 image_scores。",
        "这是当前 topk_attention 策略最关键的额外成本。若它接近 baseline 的 vision_image_encode，则说明剪枝没有节省视觉塔，只是换成可输出 attention 的路径。",
    ),
    "pruning_position_ids_prepare": (
        "剪枝前 position_ids 准备",
        "为了同步裁剪 position_ids，必要时先计算原始序列的 RoPE position ids。",
        "这是剪枝路径独有的额外成本。",
    ),
    "vispruner_total": (
        "VisPruner 裁剪模块总段",
        "WallXVisPruner 内部完整耗时，包括长度计算、分数处理、top-k、裁剪 input_ids/attention_mask/moe_token_types/position_ids、padding 和 rope_deltas。",
        "衡量剪枝操作本身是否昂贵。它不包含视觉塔 attention score 的获取。",
    ),
    "vispruner_image_lengths": (
        "图像 token 长度计算",
        "根据 image_grid_thw 和 spatial_merge_size 计算每张图对应的 LLM 视觉 token 数。",
        "通常很小。",
    ),
    "vispruner_score_prepare": (
        "剪枝分数准备",
        "将 image_scores 拉平、detach、转 float，并处理 NaN/Inf；若没有 image_scores 则回退到 image_embeds norm。",
        "通常不是主耗时。",
    ),
    "vispruner_build_keep_mask": (
        "构造保留 mask 总段",
        "包含 score_prepare 和 topk_select，输出每个视觉 token 是否保留的布尔 mask。",
        "用于判断选择逻辑本身的开销。",
    ),
    "vispruner_topk_select": (
        "逐图 top-k 选择",
        "按每张图的视觉 token 段，根据 keep_ratio 选择得分最高的 token，并保持原始顺序。",
        "如果未来换更复杂的多样性/相似度算法，该段可能明显增加。",
    ),
    "vispruner_gather_image_embeds": (
        "裁剪 image_embeds",
        "用 keep_mask 从 image_embeds 中取出保留的视觉特征。",
        "通常较小。",
    ),
    "vispruner_apply_keep_to_sequences": (
        "同步裁剪序列张量",
        "按视觉 token 保留结果，同步裁剪 input_ids、attention_mask、labels、moe_token_types、position_ids。",
        "这是硬剪枝是否正确接入主模型的关键步骤。",
    ),
    "vispruner_pad_pruned_batch": (
        "裁剪后 batch 重新 padding",
        "由于每个样本裁剪后长度可能不同，需要重新 pad input_ids、attention_mask、labels、moe_token_types、position_ids。",
        "batch size 变大或多图长度差异变大时，该段可能增加。",
    ),
    "vispruner_rope_deltas": (
        "裁剪后 rope_deltas 计算",
        "根据裁剪后的 position_ids 和 attention_mask 重新计算 mrope delta。",
        "用于保证后续生成时位置编码一致。",
    ),
    "embed_tokens": (
        "文本 token embedding",
        "将 input_ids 输入词嵌入层，得到 inputs_embeds。",
        "序列变短理论上会略降，但该段很小。",
    ),
    "scatter_image_embeds": (
        "图像特征写入 inputs_embeds",
        "把 image_embeds 写入 input_ids 中 image token 对应的位置。",
        "视觉 token 变少后通常会下降，但整体占比很小。",
    ),
    "scatter_proprioception": (
        "机器人状态 embedding 写入",
        "将 proprioception 经过投影后写入 <|propri|> 对应位置。",
        "与视觉 token 剪枝弱相关。",
    ),
    "attention_mask_to_device": (
        "attention_mask 设备对齐",
        "将 attention_mask 移到 inputs_embeds 所在设备。",
        "通常很小。",
    ),
    "position_encoding": (
        "位置编码与 MoE 分组总段",
        "包含 position_ids/RoPE 准备以及 start_indices/end_indices 等 MoE token 分组索引计算。",
        "pruned 路径中 position_ids 可能已在剪枝前准备，因此该段可能下降，不能简单理解为总位置编码成本减少。",
    ),
    "position_ids_rope": (
        "RoPE position_ids 计算",
        "根据 input_ids、image_grid_thw、attention_mask 等计算 3D mRoPE position ids。",
        "baseline 中通常在这里发生；pruned 中可能提前到 pruning_position_ids_prepare。",
    ),
    "moe_indices": (
        "MoE token 分组索引",
        "根据 moe_token_types 统计每个 expert 的 token 数，并计算 start/end indices。",
        "与视觉 token 数略相关，但本实验中影响很小。",
    ),
    "action_initialization": (
        "动作扩散初始化总段",
        "生成初始噪声、时间步、第一次 action embedding，并写入 action token 位置。",
        "主要与 action_horizon/action_dim 相关，与视觉 token 剪枝弱相关。",
    ),
    "action_init_noise": (
        "初始噪声和时间步准备",
        "生成 noisy_action 和 times/dt。",
        "与视觉剪枝无关。",
    ),
    "action_init_embed": (
        "初始 action embedding",
        "根据 timestep 和 noisy_action 生成 action token embedding。",
        "与视觉剪枝无关。",
    ),
    "scatter_action_init": (
        "初始 action embedding 写入",
        "把 action embedding 写入 inputs_embeds 中 action token 对应位置。",
        "与视觉剪枝弱相关。",
    ),
    "prefetch_forward": (
        "prefix/prefill 总段",
        "完整输入序列第一次进入主 Transformer，生成 hidden states 和 prefix KV-cache，并做第一次 action prediction。",
        "这是视觉 token 变少最可能产生收益的主模型阶段之一。",
    ),
    "prefill_transformer": (
        "prefill Transformer",
        "self.model 对完整 inputs_embeds 的第一次 forward，不含后面的 action head。",
        "如果视觉 token 剪枝有效降低主 Transformer 计算，这里应明显下降。本实验中没有明显下降。",
    ),
    "prefill_action_head": (
        "prefill action head",
        "从 action token hidden states 预测初始动作速度/动作，并更新 noisy_action。",
        "主要与 action token 数相关，与视觉 token 剪枝弱相关。",
    ),
    "cache_preprocessing": (
        "KV-cache 与 postfix 输入准备总段",
        "裁剪 prefix KV-cache，切出 postfix 输入、postfix position ids、postfix moe_token_types，并构造 postfix attention mask。",
        "是 ODE 前的准备工作。",
    ),
    "prefix_length_resolve": (
        "prefix 长度确定",
        "根据 action token 位置确定 prefix_length。",
        "通常很小。",
    ),
    "kv_cache_trim": (
        "KV-cache 裁剪",
        "将 prefill 得到的 KV-cache 裁剪到 prefix 部分，供 ODE/postfix 阶段复用。",
        "与 prefix 长度相关。",
    ),
    "postfix_slice": (
        "postfix 张量切片",
        "从完整输入中切出 action/postfix 部分。",
        "通常很小。",
    ),
    "postfix_moe_indices": (
        "postfix MoE 分组",
        "为 postfix 阶段重新计算 expert token 分组。",
        "通常很小。",
    ),
    "postfix_mask_build": (
        "postfix attention mask 构造",
        "构造 postfix query 到 prefix+postfix key 的 attention mask，并处理 padding/causal action mask。",
        "与 prefix/postfix 长度相关，但本实验中占比小。",
    ),
    "ode_integration": (
        "ODE 多步动作生成总段",
        "torchdiffeq.odeint 的 euler 多步过程。每一步会更新 action embedding，使用 prefix KV-cache 跑 postfix Transformer，再经过 action head 得到动作速度。",
        "这是本实验最大的耗时段。若它不随视觉 token 减少而下降，则端到端加速会非常有限。",
    ),
    "ode_action_embed_total": (
        "ODE 每步 action embedding 累计",
        "ODE 每个时间步中，根据当前 noisy_action 生成 action embedding 的累计耗时。",
        "timing_counts 显示本实验累计了 9 次，说明 ODE 实际执行 9 个 postfix step。",
    ),
    "ode_prepare_inputs": (
        "ODE 每步输入准备累计",
        "clone postfix_inputs_embeds，并写入当前步 action embedding。",
        "与视觉 token 剪枝弱相关。",
    ),
    "ode_transformer_total": (
        "ODE 每步 postfix Transformer 累计",
        "ODE 每个时间步中，带 prefix KV-cache 运行 postfix Transformer 的累计耗时。",
        "这是完整动作推理最大头。当前视觉剪枝主要减少 prefix 视觉 token，但 postfix 阶段主要围绕 action token 和 prefix cache，因此收益很小。",
    ),
    "ode_action_head_total": (
        "ODE 每步 action head 累计",
        "从 postfix Transformer 输出中取 action token hidden states 并投影为动作速度。",
        "主要与 action token 数相关。",
    ),
    "postprocessing": (
        "输出后处理",
        "取 ODE 最后一步动作，按需反归一化并组装输出。",
        "通常很小。",
    ),
}


ORDER = [
    "external_prepare_batch_ms",
    "total_time",
    "embed_processing",
    "image_path_total",
    "vision_image_forward",
    "image_cast",
    "vision_image_encode",
    "vision_image_encode_score",
    "pruning_position_ids_prepare",
    "vispruner_total",
    "vispruner_image_lengths",
    "vispruner_build_keep_mask",
    "vispruner_score_prepare",
    "vispruner_topk_select",
    "vispruner_gather_image_embeds",
    "vispruner_apply_keep_to_sequences",
    "vispruner_pad_pruned_batch",
    "vispruner_rope_deltas",
    "embed_tokens",
    "scatter_image_embeds",
    "scatter_proprioception",
    "attention_mask_to_device",
    "position_encoding",
    "position_ids_rope",
    "moe_indices",
    "action_initialization",
    "action_init_noise",
    "action_init_embed",
    "scatter_action_init",
    "prefetch_forward",
    "prefill_transformer",
    "prefill_action_head",
    "cache_preprocessing",
    "prefix_length_resolve",
    "kv_cache_trim",
    "postfix_slice",
    "postfix_moe_indices",
    "postfix_mask_build",
    "ode_integration",
    "ode_action_embed_total",
    "ode_prepare_inputs",
    "ode_transformer_total",
    "ode_action_head_total",
    "postprocessing",
]


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def pct(delta, base):
    return delta / base * 100.0 if base else 0.0


def get_value(summary, key):
    if key == "external_prepare_batch_ms":
        return summary.get("external_prepare_batch_ms", 0.0)
    return summary["timings_ms"].get(key, 0.0)


def row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def p(text):
    return f"<p>{text}</p>"


def h(level, text):
    return f"<h{level}>{text}</h{level}>"


def main():
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    args = data["args"]
    baseline = data["baseline_summary"]
    pruned = data["pruned_summary"]
    paired = data["paired_deltas"]
    counts = data["timing_counts"]
    all_keys = set(baseline["timings_ms"]) | set(pruned["timings_ms"]) | {
        "external_prepare_batch_ms"
    }
    ordered_keys = [key for key in ORDER if key in all_keys]
    ordered_keys += sorted(all_keys - set(ordered_keys))

    avg_before = baseline["vision_tokens_before"]
    avg_after = pruned["vision_tokens_after"]
    token_reduction = pct(avg_before - avg_after, avg_before)
    base_total = baseline["timings_ms"].get("total_time", 0.0)
    pruned_total = pruned["timings_ms"].get("total_time", 0.0)
    total_delta = pruned_total - base_total

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:'Microsoft YaHei',Arial,sans-serif;line-height:1.55;color:#222;}",
        "h1,h2,h3{color:#111;} table{border-collapse:collapse;width:100%;margin:12px 0;}",
        "th,td{border:1px solid #888;padding:6px 8px;vertical-align:top;font-size:12px;}",
        "th{background:#eee;} .note{background:#fff7d6;padding:10px;border:1px solid #d8bf60;}",
        ".good{color:#096b2e;font-weight:bold}.bad{color:#9a1b1b;font-weight:bold}.mono{font-family:Consolas,monospace;}",
        "</style></head><body>",
        h(1, "Wall-X VisPruner 30 图时间戳对比中文说明报告"),
        p("本报告根据 30 张图片的真实 baseline / VisPruner 剪枝版本运行结果重新整理，重点解释每个时间戳环节测量的含义，以及这些数据如何回答“视觉 token 减少但端到端加速很小”的问题。"),
        h(2, "1. 实验设置"),
        "<table>",
        row(["项目", "内容"], "th"),
        row(["模型", f"<span class='mono'>{html.escape(args['model_path'])}</span>"]),
        row(["图片目录", f"<span class='mono'>{html.escape(args['image_dir'])}</span>"]),
        row(["图片来源", "picsum.photos seed 图片，尺寸 640×480，用于多图压力测试而非机器人真实数据集。"]),
        row(["图片数量", str(len(data["images"]))]),
        row(["baseline", "vispruner_enable=False，strategy=original，保留全部视觉 token。"]),
        row(["pruned", f"vispruner_enable=True，strategy=topk_attention，keep_ratio={args['keep_ratio']}。"]),
        row(["warmup / iters", f"{args['warmup']} / {args['iters']}。每张图每个版本先预热，再统计多次平均。"]),
        row(["计时方式", "profile_timing=True，使用 CUDA Event；细粒度计时会引入同步开销，适合瓶颈诊断，不等同线上真实无 profiling 延迟。"]),
        "</table>",
        h(2, "2. 总体结论"),
        "<table>",
        row(["指标", "baseline", "pruned", "变化", "解释"], "th"),
        row([
            "平均视觉 token 数",
            fmt(avg_before, 2),
            fmt(avg_after, 2),
            f"-{fmt(avg_before - avg_after, 2)}（{fmt(token_reduction, 2)}%）",
            "VisPruner 已真实生效，视觉 token 约减少一半。",
        ]),
        row([
            "模型内部 total_time",
            f"{fmt(base_total)} ms",
            f"{fmt(pruned_total)} ms",
            f"{fmt(total_delta)} ms（{fmt(pct(total_delta, base_total), 2)}%）",
            "端到端动作生成几乎没有明显加速。",
        ]),
        "</table>",
        p("<b>核心判断：</b>当前 VisPruner 确实减少了视觉 token，但主要耗时集中在 ODE/postfix Transformer；同时剪枝发生在 vision tower 之后，仍需要完整视觉编码和 attention score 获取。因此 token 数下降没有显著转化为完整动作推理延迟下降。"),
        h(2, "3. 时间戳层级关系说明"),
        p("这些时间戳存在嵌套关系，不能简单全部相加。最重要的层级如下："),
        "<ul>",
        "<li><span class='mono'>total_time</span> 是模型内部总耗时。</li>",
        "<li><span class='mono'>embed_processing</span> 包含 <span class='mono'>image_path_total</span>、<span class='mono'>embed_tokens</span>、<span class='mono'>scatter_image_embeds</span> 等。</li>",
        "<li><span class='mono'>image_path_total</span> 在 baseline 中主要是 <span class='mono'>vision_image_encode</span>；在 pruned 中主要是 <span class='mono'>vision_image_encode_score</span> + <span class='mono'>pruning_position_ids_prepare</span> + <span class='mono'>vispruner_total</span>。</li>",
        "<li><span class='mono'>prefetch_forward</span> 包含 <span class='mono'>prefill_transformer</span> 和 <span class='mono'>prefill_action_head</span>。</li>",
        "<li><span class='mono'>ode_integration</span> 包含多次累计的 <span class='mono'>ode_action_embed_total</span>、<span class='mono'>ode_prepare_inputs</span>、<span class='mono'>ode_transformer_total</span>、<span class='mono'>ode_action_head_total</span>。</li>",
        "</ul>",
        h(2, "4. 各时间戳环节详细解释与前后对比"),
        "<table>",
        row(["时间戳", "测量环节", "baseline 平均", "pruned 平均", "变化", "数据意义与解释"], "th"),
    ]

    for key in ordered_keys:
        base = get_value(baseline, key)
        prun = get_value(pruned, key)
        if base == 0.0 and prun == 0.0:
            continue
        delta = prun - base
        name, meaning, interpretation = SEGMENT_DESCRIPTIONS.get(
            key,
            (key, "该时间戳为代码中记录的辅助计时段。", "用于补充定位。"),
        )
        if base == 0.0:
            change = f"{fmt(delta)} ms（仅 pruned 出现）"
        else:
            change = f"{fmt(delta)} ms（{fmt(pct(delta, base), 2)}%）"
        explanation = f"<b>{html.escape(name)}</b><br>{html.escape(meaning)}<br><br><b>解释：</b>{html.escape(interpretation)}"
        html_parts.append(
            row(
                [
                    f"<span class='mono'>{html.escape(key)}</span>",
                    html.escape(name),
                    f"{fmt(base)} ms",
                    f"{fmt(prun)} ms",
                    html.escape(change),
                    explanation,
                ]
            )
        )
    html_parts.extend(["</table>"])

    html_parts.extend(
        [
            h(2, "5. 与“加速很小”问题直接相关的关键证据"),
            "<ol>",
            f"<li><b>视觉 token 确实减少：</b>平均从 {fmt(avg_before, 2)} 降到 {fmt(avg_after, 2)}，下降 {fmt(token_reduction, 2)}%。这说明剪枝逻辑本身生效。</li>",
            f"<li><b>图像路径没有变快：</b><span class='mono'>vision_image_forward</span> 从 {fmt(get_value(baseline, 'vision_image_forward'))} ms 到 {fmt(get_value(pruned, 'vision_image_forward'))} ms，反而略慢。原因是 pruned 需要 <span class='mono'>vision_image_encode_score</span> 获取 attention score，还要执行 <span class='mono'>vispruner_total</span>。</li>",
            f"<li><b>prefill Transformer 没有明显收益：</b><span class='mono'>prefill_transformer</span> 从 {fmt(get_value(baseline, 'prefill_transformer'))} ms 到 {fmt(get_value(pruned, 'prefill_transformer'))} ms。本轮测试没有体现出视觉 token 缩短带来的 prefix 主模型加速。</li>",
            f"<li><b>最大耗时段几乎不变：</b><span class='mono'>ode_transformer_total</span> 从 {fmt(get_value(baseline, 'ode_transformer_total'))} ms 到 {fmt(get_value(pruned, 'ode_transformer_total'))} ms。完整推理大部分时间花在 ODE/postfix Transformer，这部分主要围绕 action token 和 prefix cache 运行，因此对视觉 token 裁剪不敏感。</li>",
            "</ol>",
        ]
    )

    html_parts.extend(
        [
            h(2, "6. 逐图 paired total_time 结果"),
            "<table>",
            row(["序号", "图片", "token before", "token after", "token 下降", "baseline total", "pruned total", "变化"], "th"),
        ]
    )
    for item in paired:
        image_name = Path(item["image_path"]).name
        html_parts.append(
            row(
                [
                    str(item["image_index"]),
                    f"<span class='mono'>{html.escape(image_name)}</span>",
                    str(item["vision_tokens_before"]),
                    str(item["vision_tokens_after"]),
                    f"{fmt(item['token_reduction_pct'], 2)}%",
                    f"{fmt(item['baseline_total_ms'])} ms",
                    f"{fmt(item['pruned_total_ms'])} ms",
                    f"{fmt(item['total_delta_ms'])} ms（{fmt(item['total_delta_pct'], 2)}%）",
                ]
            )
        )
    html_parts.extend(["</table>"])

    html_parts.extend(
        [
            h(2, "7. timing_counts 说明"),
            p("timing_counts 表示某个时间戳在一次被统计的模型调用中累计了多少次。多数阶段为 1；ODE 内部阶段为 9，说明本实验中 ODE euler 实际执行了 9 个 postfix Transformer step。"),
            "<table>",
            row(["版本", "时间戳", "count"], "th"),
        ]
    )
    for case_name, case_counts in sorted(counts.items()):
        for key, value in sorted(case_counts.items()):
            if key.startswith("ode_") or key in {"total_time", "vision_image_forward", "prefill_transformer"}:
                html_parts.append(row([case_name, f"<span class='mono'>{html.escape(key)}</span>", str(value)]))
    html_parts.extend(["</table>"])

    html_parts.extend(
        [
            h(2, "8. 实验局限与下一步建议"),
            "<ul>",
            "<li>本报告使用 picsum 随机图片，不是机器人真实任务帧；它适合性能诊断，不代表任务成功率或真实数据分布。</li>",
            "<li>所有细粒度时间戳都在 profile_timing=True 下测量，会引入 CUDA 同步开销。因此应看 baseline/pruned 的相对差异，而不是把绝对时间当作线上延迟。</li>",
            "<li>当前 topk_attention 策略需要先完整运行视觉塔并输出 attention score。若想得到明显端到端加速，下一步应尝试更低成本的打分方式，或在 vision tower 更早阶段剪枝。</li>",
            "<li>若目标是动作推理端到端加速，还需要分析 ODE/postfix Transformer：减少 ODE step、优化 postfix attention、缓存 action-invariant 部分，可能比单纯视觉 token 剪枝更直接。</li>",
            "</ul>",
            h(2, "9. 原始结果文件"),
            p(f"结构化 JSON 原始结果：<span class='mono'>{html.escape(str(RESULTS_JSON))}</span>"),
            "</body></html>",
        ]
    )

    REPORT_DOC.write_text("\n".join(html_parts), encoding="utf-8")
    print(REPORT_DOC)


if __name__ == "__main__":
    main()
