import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from wall_x.model.vispruner_score_predictor import load_token_score_predictor


def resolve_feature_key(record: dict, requested_key: str) -> str:
    if requested_key != "auto":
        return requested_key
    if "predictor_features" in record:
        return "predictor_features"
    return "image_embeds"


def make_topk_mask(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    keep_count = max(0, min(int(keep_count), scores.numel()))
    mask = torch.zeros(scores.shape[0], dtype=torch.bool)
    if keep_count == 0:
        return mask
    keep = torch.topk(scores.float(), k=keep_count, largest=True).indices
    mask[keep] = True
    return mask


def resolve_teacher_paths(path: str) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return [Path(item["path"]) for item in manifest["shards"]]
        return sorted(root.glob("teacher_scores_shard_*.pt"))
    return [root]


@torch.no_grad()
def evaluate(args):
    teacher_paths = resolve_teacher_paths(args.teacher_path)
    if not teacher_paths:
        raise ValueError(f"No teacher records found under {args.teacher_path}")
    first_payload = torch.load(teacher_paths[0], map_location="cpu")
    first_record = first_payload["records"][0]
    feature_key = resolve_feature_key(first_record, args.feature_key)
    input_dim = int(first_record[feature_key].shape[-1])

    predictor = load_token_score_predictor(
        args.checkpoint,
        default_input_dim=input_dim,
        strict=True,
    ).to(args.device)
    predictor.eval()

    losses = []
    overlaps = []
    agreements = []
    records_seen = 0
    for path_idx, teacher_path in enumerate(teacher_paths, start=1):
        if args.max_shards > 0 and path_idx > args.max_shards:
            break
        payload = torch.load(teacher_path, map_location="cpu")
        for record in payload["records"]:
            record_feature_key = resolve_feature_key(record, args.feature_key)
            features = record[record_feature_key].float().to(args.device)
            teacher_mask = record["keep_mask"].bool().reshape(-1)
            teacher_scores = record["image_scores"].float().reshape(-1)
            teacher_scores = (teacher_scores - teacher_scores.mean()) / (
                teacher_scores.std(unbiased=False) + 1e-6
            )

            pred_scores = predictor(features).detach().float().cpu().reshape(-1)
            if pred_scores.shape[0] != teacher_scores.shape[0]:
                raise ValueError(
                    f"Predictor/teacher length mismatch for dataset_index="
                    f"{record.get('dataset_index')}: "
                    f"{pred_scores.shape[0]} vs {teacher_scores.shape[0]}"
                )

            pred_mask = make_topk_mask(pred_scores, int(teacher_mask.sum().item()))
            overlap = (pred_mask & teacher_mask).sum().item() / max(
                1, int(teacher_mask.sum().item())
            )
            agreement = (pred_mask == teacher_mask).float().mean().item()
            loss = F.mse_loss(pred_scores, teacher_scores).item()

            losses.append(loss)
            overlaps.append(overlap)
            agreements.append(agreement)
            records_seen += 1
        if args.log_every_shards > 0 and (
            path_idx == 1
            or path_idx == len(teacher_paths)
            or path_idx % args.log_every_shards == 0
        ):
            print(
                f"[EVAL] shard={path_idx}/{len(teacher_paths)} records={records_seen}",
                flush=True,
            )

    print(
        {
            "records": records_seen,
            "shards": min(
                len(teacher_paths),
                args.max_shards if args.max_shards > 0 else len(teacher_paths),
            ),
            "feature_key": feature_key,
            "mean_score_mse": sum(losses) / len(losses),
            "mean_topk_overlap": sum(overlaps) / len(overlaps),
            "mean_mask_agreement": sum(agreements) / len(agreements),
        },
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a VisPruner score predictor against teacher scores/masks."
    )
    parser.add_argument(
        "--teacher-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/libero_teacher_scores.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--feature-key",
        default="auto",
        help="Feature tensor key in teacher records. Use auto for predictor_features then image_embeds.",
    )
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--log-every-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
