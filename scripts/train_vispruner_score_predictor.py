import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wall_x.model.vispruner_score_predictor import TokenScorePredictor


def resolve_feature_key(record: dict, requested_key: str) -> str:
    if requested_key != "auto":
        return requested_key
    if "predictor_features" in record:
        return "predictor_features"
    return "image_embeds"


def resolve_teacher_paths(path: str) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return [Path(item["path"]) for item in manifest["shards"]]
        return sorted(root.glob("teacher_scores_shard_*.pt"))
    return [root]


def tensors_from_records(records, feature_key: str, target: str):
    features = []
    targets = []
    for record in records:
        resolved_feature_key = resolve_feature_key(record, feature_key)
        if resolved_feature_key not in record:
            raise ValueError(
                f"Teacher file does not contain {resolved_feature_key}. Re-run "
                "collect_vispruner_teacher_scores.py with a matching --feature-source."
            )
        feature = record[resolved_feature_key].float()
        if target == "score":
            if record["image_scores"] is None:
                raise ValueError(
                    f"Teacher record {record.get('dataset_index')} has no image_scores."
                )
            teacher_target = record["image_scores"].float().reshape(-1)
            teacher_target = (teacher_target - teacher_target.mean()) / (
                teacher_target.std(unbiased=False) + 1e-6
            )
        elif target == "mask":
            teacher_target = record["keep_mask"].float().reshape(-1)
        else:
            raise ValueError(f"Unsupported target: {target}")

        if feature.shape[0] != teacher_target.shape[0]:
            raise ValueError(
                f"{resolved_feature_key}/{target} length mismatch in record "
                f"{record.get('dataset_index')}: "
                f"{feature.shape[0]} vs {teacher_target.shape[0]}"
            )
        features.append(feature)
        targets.append(teacher_target)
    return (
        torch.cat(features, dim=0),
        torch.cat(targets, dim=0),
    )


def load_teacher_tensors(path: str, feature_key: str, target: str):
    payload = torch.load(path, map_location="cpu")
    features, targets = tensors_from_records(payload["records"], feature_key, target)
    return features, targets, payload.get("meta", {})


def split_dataset(features, targets, val_ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(features.shape[0], generator=generator)
    val_size = max(1, int(features.shape[0] * val_ratio)) if val_ratio > 0 else 0
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_set = TensorDataset(features[train_idx], targets[train_idx])
    val_set = TensorDataset(features[val_idx], targets[val_idx]) if val_size > 0 else None
    return train_set, val_set


def predictor_loss(pred, target, loss_type: str):
    if loss_type == "mask":
        return F.binary_cross_entropy_with_logits(pred.float(), target.float())
    return F.mse_loss(pred.float(), target.float())


def evaluate(model, loader, device, loss_type: str):
    if loader is None:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for xs, ys in loader:
            xs = xs.to(device).float()
            ys = ys.to(device)
            pred = model(xs)
            losses.append(predictor_loss(pred, ys, loss_type).item())
    return sum(losses) / len(losses) if losses else None


def build_model(args, input_dim: int):
    model = TokenScorePredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    return model, optimizer


def train_one_loader(model, optimizer, loader, args):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for xs, ys in loader:
        xs = xs.to(args.device).float()
        ys = ys.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xs)
        loss = predictor_loss(pred, ys, args.target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss, total_batches


def train_in_memory(args):
    features, targets, teacher_meta = load_teacher_tensors(
        args.teacher_path, args.feature_key, args.target
    )
    train_set, val_set = split_dataset(features, targets, args.val_ratio, args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_set is not None
        else None
    )

    model, optimizer = build_model(args, features.shape[-1])

    best_val = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss_sum, batch_count = train_one_loader(model, optimizer, train_loader, args)
        train_loss = loss_sum / max(1, batch_count)
        val_loss = evaluate(model, val_loader, args.device, args.target)
        metric = val_loss if val_loss is not None else train_loss
        if best_val is None or metric < best_val:
            best_val = metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[PREDICTOR] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}" if val_loss is not None else
            f"[PREDICTOR] epoch={epoch} train_loss={train_loss:.6f}",
            flush=True,
        )

    return model, best_state, best_val, teacher_meta, int(features.shape[0])


def evaluate_shards(model, shard_paths, args):
    if not shard_paths:
        return None
    losses = []
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu")
        features, targets = tensors_from_records(
            payload["records"], args.feature_key, args.target
        )
        loader = DataLoader(
            TensorDataset(features, targets),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        loss = evaluate(model, loader, args.device, args.target)
        if loss is not None:
            losses.append(loss)
    return sum(losses) / len(losses) if losses else None


def train_from_shards(args, shard_paths):
    first_payload = torch.load(shard_paths[0], map_location="cpu")
    first_features, _ = tensors_from_records(
        first_payload["records"], args.feature_key, args.target
    )
    teacher_meta = first_payload.get("meta", {})
    input_dim = int(first_features.shape[-1])
    del first_features

    val_count = int(len(shard_paths) * args.val_ratio)
    if args.val_ratio > 0:
        val_count = max(1, val_count)
    val_paths = shard_paths[-val_count:] if val_count > 0 else []
    train_paths = shard_paths[:-val_count] if val_count > 0 else shard_paths
    if not train_paths:
        train_paths = shard_paths
        val_paths = []

    model, optimizer = build_model(args, input_dim)
    best_val = None
    best_state = None
    num_tokens = 0
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_paths), generator=generator).tolist()
        loss_sum = 0.0
        batch_count = 0
        epoch_tokens = 0
        for step_idx, idx in enumerate(order, start=1):
            payload = torch.load(train_paths[idx], map_location="cpu")
            features, targets = tensors_from_records(
                payload["records"], args.feature_key, args.target
            )
            shard_tokens = int(features.shape[0])
            epoch_tokens += shard_tokens
            loader = DataLoader(
                TensorDataset(features, targets),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            shard_loss, shard_batches = train_one_loader(model, optimizer, loader, args)
            loss_sum += shard_loss
            batch_count += shard_batches
            if args.log_every_shards > 0 and (
                step_idx == 1
                or step_idx == len(order)
                or step_idx % args.log_every_shards == 0
            ):
                running_loss = loss_sum / max(1, batch_count)
                print(
                    f"[PREDICTOR] epoch={epoch} shard={step_idx}/{len(order)} "
                    f"shard_tokens={shard_tokens} epoch_tokens={epoch_tokens} "
                    f"running_train_loss={running_loss:.6f}",
                    flush=True,
                )
        num_tokens = max(num_tokens, epoch_tokens)
        train_loss = loss_sum / max(1, batch_count)
        val_loss = evaluate_shards(model, val_paths[: args.max_val_shards], args)
        metric = val_loss if val_loss is not None else train_loss
        if best_val is None or metric < best_val:
            best_val = metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[PREDICTOR] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}" if val_loss is not None else
            f"[PREDICTOR] epoch={epoch} train_loss={train_loss:.6f}",
            flush=True,
        )

    return model, best_state, best_val, teacher_meta, num_tokens


def train(args):
    torch.manual_seed(args.seed)
    shard_paths = resolve_teacher_paths(args.teacher_path)
    if not shard_paths:
        raise ValueError(f"No teacher shards found under {args.teacher_path}")
    if len(shard_paths) == 1 and not Path(args.teacher_path).is_dir():
        model, best_state, best_val, teacher_meta, num_tokens = train_in_memory(args)
    else:
        model, best_state, best_val, teacher_meta, num_tokens = train_from_shards(
            args, shard_paths
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "config": model.config_dict(),
            "meta": {
                "teacher_path": args.teacher_path,
                "teacher_meta": teacher_meta,
                "feature_key": args.feature_key,
                "target": args.target,
                "num_tokens": int(num_tokens),
                "best_loss": best_val,
            },
        },
        output_path,
    )
    print(f"[PREDICTOR] saved checkpoint to {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small MLP predictor to imitate VisPruner teacher scores."
    )
    parser.add_argument(
        "--teacher-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/libero_teacher_scores.pt",
    )
    parser.add_argument(
        "--output-path",
        default="/root/autodl-tmp/wall_x/workspace/vispruner_teacher_scores/token_score_predictor.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--feature-key",
        default="auto",
        help="Feature tensor key in teacher records. Use auto for predictor_features then image_embeds.",
    )
    parser.add_argument(
        "--target",
        default="score",
        choices=["score", "mask"],
        help="Train against normalized teacher scores or binary keep_mask.",
    )
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-val-shards",
        type=int,
        default=4,
        help="Maximum number of validation shards evaluated per epoch for sharded training.",
    )
    parser.add_argument(
        "--log-every-shards",
        type=int,
        default=1,
        help="For sharded training, print progress every N shards.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
