from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class VisualPruneResult:
    image_embeds: torch.Tensor
    input_ids: torch.LongTensor
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.LongTensor] = None
    moe_token_types: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    rope_deltas: Optional[torch.LongTensor] = None
    keep_indices: Optional[List[torch.Tensor]] = None
    pruned: bool = False


class WallXVisPruner(nn.Module):
    SCORE_BASED_STRATEGIES = {
        "topk_attention",
        "predictor_score",
        "predictor_early",
        "norm",
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.strategy = str(getattr(config, "vispruner_strategy", "original"))
        self.keep_ratio = float(getattr(config, "vispruner_keep_ratio", 1.0))
        self.min_tokens = int(getattr(config, "vispruner_min_tokens", 1))

    @property
    def enabled(self) -> bool:
        return (
            bool(getattr(self.config, "vispruner_enable", False))
            and self.strategy != "original"
            and self.keep_ratio < 1.0
        )

    def _image_token_lengths(
        self, image_grid_thw: torch.LongTensor, spatial_merge_size: int
    ) -> torch.LongTensor:
        lengths = (
            image_grid_thw[:, 0]
            * (image_grid_thw[:, 1] // spatial_merge_size)
            * (image_grid_thw[:, 2] // spatial_merge_size)
        )
        return lengths.to(dtype=torch.long)

    def _pad_1d(self, values: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(v.shape[0] for v in values)
        out = values[0].new_full((len(values), max_len), pad_value)
        for i, value in enumerate(values):
            out[i, : value.shape[0]] = value
        return out

    def _pad_position_ids(
        self, values: List[torch.Tensor], pad_value: int = 1
    ) -> torch.Tensor:
        max_len = max(v.shape[-1] for v in values)
        if values[0].dim() == 2:
            out = values[0].new_full(
                (values[0].shape[0], len(values), max_len), pad_value
            )
            for i, value in enumerate(values):
                out[:, i, : value.shape[-1]] = value
            return out

        out = values[0].new_full((len(values), max_len), pad_value)
        for i, value in enumerate(values):
            out[i, : value.shape[-1]] = value
        return out

    def _compute_rope_deltas(
        self,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if position_ids is None or attention_mask is None:
            return None

        deltas = []
        for batch_idx in range(attention_mask.shape[0]):
            active = attention_mask[batch_idx].bool()
            if not active.any():
                deltas.append(
                    torch.zeros(
                        (), dtype=position_ids.dtype, device=position_ids.device
                    )
                )
                continue
            if position_ids.dim() == 3:
                max_position = position_ids[:, batch_idx, active].max()
            else:
                max_position = position_ids[batch_idx, active].max()
            deltas.append(max_position + 1 - active.sum().to(position_ids.device))
        return torch.stack(deltas).unsqueeze(1)

    def _build_image_keep_mask(
        self,
        image_embeds: torch.Tensor,
        image_scores: Optional[torch.Tensor],
        image_lengths: torch.LongTensor,
        perf_timer=None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        if self.strategy not in self.SCORE_BASED_STRATEGIES:
            raise ValueError(
                f"Unsupported vispruner strategy: {self.strategy}. "
                "Supported strategies are 'original', 'topk_attention', "
                "'predictor_score', 'predictor_early', and 'norm'."
            )

        total_tokens = int(image_lengths.sum().item())
        if total_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"Image grid token count ({total_tokens}) does not match image_embeds ({image_embeds.shape[0]})."
            )

        if perf_timer is not None:
            perf_timer.start("vispruner_score_prepare")
        if self.strategy == "norm":
            scores = image_embeds.detach().float().norm(dim=-1)
        elif image_scores is None:
            if self.strategy == "predictor_score":
                raise ValueError(
                    "vispruner strategy 'predictor_score' requires predictor scores, "
                    "but image_scores is None."
                )
            scores = image_embeds.detach().float().norm(dim=-1)
        else:
            scores = image_scores.detach().float().reshape(-1).to(image_embeds.device)
            if scores.shape[0] != image_embeds.shape[0]:
                raise ValueError(
                    f"Image score count ({scores.shape[0]}) does not match image_embeds ({image_embeds.shape[0]})."
                )
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if perf_timer is not None:
            perf_timer.stop("vispruner_score_prepare")

        keep_mask = torch.zeros(
            image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device
        )
        keep_indices: List[torch.Tensor] = []
        offset = 0
        if perf_timer is not None:
            perf_timer.start("vispruner_topk_select")
        for length_tensor in image_lengths:
            length = int(length_tensor.item())
            keep_count = int(torch.ceil(torch.tensor(length * self.keep_ratio)).item())
            keep_count = max(self.min_tokens, keep_count)
            keep_count = min(length, keep_count)

            local_scores = scores[offset : offset + length]
            local_keep = torch.topk(local_scores, k=keep_count, largest=True).indices
            local_keep = local_keep.sort().values
            keep_mask[offset + local_keep] = True
            keep_indices.append(local_keep)
            offset += length
        if perf_timer is not None:
            perf_timer.stop("vispruner_topk_select")

        return keep_mask, keep_indices

    def forward(
        self,
        image_embeds: torch.Tensor,
        image_scores: Optional[torch.Tensor],
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        image_token_id: int,
        spatial_merge_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pad_token_id: int = 0,
        label_ignore_index: int = -100,
        perf_timer=None,
        image_keep_mask: Optional[torch.Tensor] = None,
    ) -> VisualPruneResult:
        if not self.enabled:
            return VisualPruneResult(
                image_embeds=image_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                moe_token_types=moe_token_types,
                position_ids=position_ids,
                pruned=False,
            )

        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "WallXVisPruner only supports 2D attention_mask when hard-pruning image tokens."
            )

        if perf_timer is not None:
            perf_timer.start("vispruner_total")
            perf_timer.start("vispruner_image_lengths")
        image_lengths = self._image_token_lengths(
            image_grid_thw, spatial_merge_size
        ).to(input_ids.device)
        original_image_token_count = int(image_lengths.sum().item())
        if perf_timer is not None:
            perf_timer.stop("vispruner_image_lengths")
            perf_timer.start("vispruner_build_keep_mask")
        if image_keep_mask is None:
            image_keep_mask, keep_indices = self._build_image_keep_mask(
                image_embeds, image_scores, image_lengths, perf_timer=perf_timer
            )
        else:
            image_keep_mask = image_keep_mask.to(
                device=input_ids.device, dtype=torch.bool
            ).reshape(-1)
            if image_keep_mask.shape[0] != original_image_token_count:
                raise ValueError(
                    f"Image keep mask count ({image_keep_mask.shape[0]}) does not "
                    f"match image grid token count ({original_image_token_count})."
                )
            keep_indices = []
            offset = 0
            for length_tensor in image_lengths:
                length = int(length_tensor.item())
                local_keep = (
                    image_keep_mask[offset : offset + length]
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                keep_indices.append(local_keep)
                offset += length
        if perf_timer is not None:
            perf_timer.stop("vispruner_build_keep_mask")
            perf_timer.start("vispruner_gather_image_embeds")
        image_keep_mask_for_embeds = image_keep_mask.to(image_embeds.device)
        if image_embeds.shape[0] == image_keep_mask_for_embeds.shape[0]:
            pruned_image_embeds = image_embeds[image_keep_mask_for_embeds]
        elif image_embeds.shape[0] == int(image_keep_mask_for_embeds.sum().item()):
            pruned_image_embeds = image_embeds
        else:
            raise ValueError(
                f"image_embeds has {image_embeds.shape[0]} rows, but keep mask has "
                f"{image_keep_mask_for_embeds.shape[0]} tokens and "
                f"{int(image_keep_mask_for_embeds.sum().item())} kept tokens."
            )
        if perf_timer is not None:
            perf_timer.stop("vispruner_gather_image_embeds")

        if perf_timer is not None:
            perf_timer.start("vispruner_apply_keep_to_sequences")
        new_input_ids = []
        new_attention_mask = [] if attention_mask is not None else None
        new_labels = [] if labels is not None else None
        new_moe_token_types = [] if moe_token_types is not None else None
        new_position_ids = [] if position_ids is not None else None

        image_offset = 0
        grid_offset = 0
        for batch_idx in range(input_ids.shape[0]):
            image_positions = (
                (input_ids[batch_idx] == image_token_id)
                .nonzero(as_tuple=False)
                .flatten()
            )
            expected = image_positions.numel()
            consumed = 0
            sample_image_keep = []

            while consumed < expected:
                if grid_offset >= image_lengths.shape[0]:
                    raise ValueError(
                        "Not enough image_grid_thw rows for the image tokens in input_ids."
                    )
                length = int(image_lengths[grid_offset].item())
                sample_image_keep.append(
                    image_keep_mask[image_offset : image_offset + length]
                )
                image_offset += length
                grid_offset += 1
                consumed += length

            if consumed != expected:
                raise ValueError(
                    f"Image token count in sample {batch_idx} ({expected}) does not match image_grid_thw chunks ({consumed})."
                )

            seq_keep = torch.ones(
                input_ids.shape[1], dtype=torch.bool, device=input_ids.device
            )
            if expected > 0:
                seq_keep[image_positions] = torch.cat(sample_image_keep).to(
                    input_ids.device
                )

            new_input_ids.append(input_ids[batch_idx, seq_keep])
            if attention_mask is not None:
                new_attention_mask.append(attention_mask[batch_idx, seq_keep])
            if labels is not None:
                new_labels.append(labels[batch_idx, seq_keep])
            if moe_token_types is not None:
                new_moe_token_types.append(moe_token_types[batch_idx, seq_keep])
            if position_ids is not None:
                if position_ids.dim() == 3:
                    new_position_ids.append(position_ids[:, batch_idx, seq_keep])
                else:
                    new_position_ids.append(position_ids[batch_idx, seq_keep])
        if perf_timer is not None:
            perf_timer.stop("vispruner_apply_keep_to_sequences")

        if (
            image_offset != original_image_token_count
            or grid_offset != image_lengths.shape[0]
        ):
            raise ValueError(
                "Unused image features remained after pruning; check batch/image ordering."
            )

        if perf_timer is not None:
            perf_timer.start("vispruner_pad_pruned_batch")
        pruned_input_ids = self._pad_1d(new_input_ids, pad_token_id)
        pruned_attention_mask = (
            self._pad_1d(new_attention_mask, 0)
            if new_attention_mask is not None
            else None
        )
        pruned_labels = (
            self._pad_1d(new_labels, label_ignore_index)
            if new_labels is not None
            else None
        )
        pruned_moe_token_types = (
            self._pad_1d(new_moe_token_types, 0)
            if new_moe_token_types is not None
            else None
        )
        pruned_position_ids = (
            self._pad_position_ids(new_position_ids)
            if new_position_ids is not None
            else None
        )
        if perf_timer is not None:
            perf_timer.stop("vispruner_pad_pruned_batch")
            perf_timer.start("vispruner_rope_deltas")
        rope_deltas = self._compute_rope_deltas(
            pruned_position_ids, pruned_attention_mask
        )
        if perf_timer is not None:
            perf_timer.stop("vispruner_rope_deltas")
            perf_timer.stop("vispruner_total")

        return VisualPruneResult(
            image_embeds=pruned_image_embeds,
            input_ids=pruned_input_ids,
            attention_mask=pruned_attention_mask,
            labels=pruned_labels,
            moe_token_types=pruned_moe_token_types,
            position_ids=pruned_position_ids,
            rope_deltas=rope_deltas,
            keep_indices=keep_indices,
            pruned=True,
        )
