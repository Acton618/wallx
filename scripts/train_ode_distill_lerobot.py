import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.profile_vispruner_lerobot_media import (  # noqa: E402
    build_train_config,
    make_image_batch,
    sync_if_cuda,
)
from wall_x.data.utils import KEY_MAPPINGS  # noqa: E402
from wall_x.model.ode_distill_utils import (  # noqa: E402
    action_preprocessor_state_dict,
    freeze_for_ode_distill,
    get_ode_distill_config,
    trainable_parameter_names,
)
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (  # noqa: E402
    Qwen2_5_VLMoEForAction,
)


class ArgsNamespace(argparse.Namespace):
    pass


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wallx_model(args, enable_pruning: bool, train: bool = False):
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        args.model_path,
        train_config=build_train_config(args, enable_pruning),
    )
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        model.to_bfloat16_for_selected_params()
    model.train(train)
    return model


def make_items(args, total_count: int):
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    image_key = args.image_key
    if image_key is None:
        image_key = next(iter(KEY_MAPPINGS[args.repo_id]["camera"].keys()))

    if total_count > len(dataset):
        raise ValueError(f"Requested {total_count} samples but dataset only has {len(dataset)}")
    if args.sample_strategy == "linspace":
        indices = np.linspace(0, max(0, len(dataset) - 1), total_count, dtype=int)
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=total_count, replace=False)
    items = []
    for local_idx, dataset_idx in enumerate(indices.tolist()):
        sample = dataset[int(dataset_idx)]
        items.append(
            {
                "split": "train" if local_idx < args.train_samples else "val",
                "local_index": local_idx,
                "dataset_index": int(dataset_idx),
                "sample": sample,
                "image_key": image_key,
                "source": f"dataset_index={int(dataset_idx)} image_key={image_key}",
            }
        )
    return items


def flow_call_kwargs(batch, args, num_inference_timesteps: int, profile_timing: bool = False):
    return dict(
        **batch,
        action_dim=args.action_dim,
        action_horizon=args.pred_horizon,
        mode="predict",
        predict_mode="diffusion",
        unnorm=False,
        num_inference_timesteps=num_inference_timesteps,
        profile_timing=profile_timing,
        print_timing=False,
    )


def flow_generate_kwargs(batch, args, num_inference_timesteps: int, profile_timing: bool = False):
    kwargs = flow_call_kwargs(batch, args, num_inference_timesteps, profile_timing)
    kwargs.pop("mode", None)
    kwargs.pop("predict_mode", None)
    return kwargs


def call_flow_with_grad(model, batch, args, num_inference_timesteps: int):
    fn = getattr(model.generate_flow_action, "__wrapped__", None)
    if fn is None:
        raise RuntimeError("generate_flow_action has no __wrapped__; cannot bypass no_grad for distillation")
    return fn(model, **flow_generate_kwargs(batch, args, num_inference_timesteps, False))


@torch.no_grad()
def build_teacher_cache(args, out_dir: Path, items):
    cache_path = out_dir / "teacher_labels.pt"
    meta_path = out_dir / "teacher_labels_meta.json"
    if cache_path.exists() and not args.regenerate_teacher:
        print(f"[ODE_DISTILL] reuse teacher cache: {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu"), json.loads(meta_path.read_text())

    teacher = load_wallx_model(args, args.enable_pruning, train=False)
    teacher.eval()
    labels = []
    meta = []
    start = time.perf_counter()
    for idx, item in enumerate(items, start=1):
        batch = make_image_batch(teacher, item["sample"], item["image_key"], args)
        set_seed(args.teacher_seed + item["local_index"], args.device)
        out = teacher(**flow_call_kwargs(batch, args, args.teacher_num_inference_timesteps))
        action = out["predict_action"].detach().cpu().to(torch.float32).squeeze(0)
        labels.append(action)
        meta.append(
            {
                "split": item["split"],
                "local_index": item["local_index"],
                "dataset_index": item["dataset_index"],
                "source": item["source"],
            }
        )
        if idx % args.log_every == 0 or idx == len(items):
            print(f"[ODE_DISTILL][teacher] {idx}/{len(items)}", flush=True)
    teacher_labels = torch.stack(labels)
    payload = {"teacher_actions": teacher_labels}
    torch.save(payload, cache_path)
    meta_obj = {
        "items": meta,
        "teacher_num_inference_timesteps": args.teacher_num_inference_timesteps,
        "elapsed_sec": time.perf_counter() - start,
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    del teacher
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload, meta_obj


def prepare_student(args):
    student = load_wallx_model(args, args.enable_pruning, train=True)
    lora_cfg = args.ode_distill["lora"]
    if lora_cfg.get("enable", True):
        student.add_lora(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        )
    freeze_for_ode_distill(student)
    student.train()
    names = trainable_parameter_names(student)
    print(f"[ODE_DISTILL] trainable tensors: {len(names)}", flush=True)
    for name in names[:30]:
        print(f"[ODE_DISTILL] trainable {name}", flush=True)
    if len(names) > 30:
        print(f"[ODE_DISTILL] ... {len(names) - 30} more trainable tensors", flush=True)
    return student


def masked_smooth_l1(student_action, teacher_action, dof_mask):
    loss = F.smooth_l1_loss(student_action, teacher_action, reduction="none")
    mask = dof_mask.to(loss.device, loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def eval_student(args, student, items, teacher_actions, indices, split_name: str):
    student.eval()
    rows = []
    timing_dicts = []
    counts = {}
    with torch.no_grad():
        for eval_i, item_idx in enumerate(indices):
            item = items[item_idx]
            batch = make_image_batch(student, item["sample"], item["image_key"], args)
            set_seed(args.student_seed + item["local_index"], args.device)
            profile = eval_i < args.eval_timing_samples
            out = student(
                **flow_call_kwargs(
                    batch,
                    args,
                    args.student_num_inference_timesteps,
                    profile_timing=profile,
                )
            )
            if profile:
                timing_dicts.append(dict(out.get("timing_results_ms", {})))
                counts = dict(out.get("timing_counts", counts))
            pred = out["predict_action"].detach().cpu().to(torch.float32).squeeze(0)
            target = teacher_actions[item_idx]
            diff = pred - target
            rows.append(
                {
                    "local_index": item["local_index"],
                    "dataset_index": item["dataset_index"],
                    "mae": float(diff.abs().mean().item()),
                    "rmse": float(torch.sqrt((diff**2).mean()).item()),
                    "max_abs": float(diff.abs().max().item()),
                    "endpoint_mae": float(diff[-1].abs().mean().item()),
                }
            )
    student.train()
    avg_timing = {}
    if timing_dicts:
        keys = sorted({key for td in timing_dicts for key in td})
        avg_timing = {key: sum(td.get(key, 0.0) for td in timing_dicts) / len(timing_dicts) for key in keys}
    summary = {
        "split": split_name,
        "num_samples": len(rows),
        "mae": float(np.mean([row["mae"] for row in rows])) if rows else 0.0,
        "rmse": float(np.mean([row["rmse"] for row in rows])) if rows else 0.0,
        "max_abs": float(np.mean([row["max_abs"] for row in rows])) if rows else 0.0,
        "endpoint_mae": float(np.mean([row["endpoint_mae"] for row in rows])) if rows else 0.0,
        "timings_ms": avg_timing,
        "timing_counts": counts,
        "rows": rows,
    }
    return summary


def save_checkpoint(args, out_dir: Path, student, metrics, distill_config):
    if hasattr(student.model, "save_pretrained") and args.ode_distill["lora"].get("enable", True):
        student.model.save_pretrained(out_dir)
    save_file(action_preprocessor_state_dict(student), str(out_dir / "action_modules.safetensors"))
    (out_dir / "distill_config.yml").write_text(
        yaml.safe_dump(distill_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trainer_state = {
        "epoch": metrics.get("epoch"),
        "global_step": metrics.get("global_step"),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
    }
    (out_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_eval_report(args, out_dir: Path, metrics):
    val = metrics["val"]
    train = metrics.get("train_eval", {})
    timing = val.get("timings_ms", {})
    lines = [
        "# Wall-X ODE Distillation Report\n",
        f"- model_path: `{args.model_path}`",
        f"- dataset_root: `{args.dataset_root}`",
        f"- repo_id: `{args.repo_id}`",
        f"- image_key: `{args.image_key}`",
        f"- train_samples: `{args.train_samples}`",
        f"- val_samples: `{args.val_samples}`",
        f"- teacher_num_inference_timesteps: `{args.teacher_num_inference_timesteps}`",
        f"- student_num_inference_timesteps: `{args.student_num_inference_timesteps}`",
        f"- epochs: `{args.epochs}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- vispruner_enable: `{args.enable_pruning}`",
        "",
        "## Summary\n",
        f"- train_eval MAE: `{train.get('mae', 0.0):.6f}`",
        f"- val MAE: `{val.get('mae', 0.0):.6f}`",
        f"- val RMSE: `{val.get('rmse', 0.0):.6f}`",
        f"- val max_abs: `{val.get('max_abs', 0.0):.6f}`",
        f"- val endpoint_mae: `{val.get('endpoint_mae', 0.0):.6f}`",
        "",
        "## Student Timing Samples\n",
        "| stage | ms |",
        "|---|---:|",
    ]
    for key in ["total_time", "prefetch_forward", "ode_integration", "ode_transformer_total", "postprocessing"]:
        if key in timing:
            lines.append(f"| `{key}` | {timing[key]:.3f} |")
    lines.extend([
        "",
        "## Output Files\n",
        f"- `{out_dir / 'distill_config.yml'}`",
        f"- `{out_dir / 'adapter_model.safetensors'}`",
        f"- `{out_dir / 'action_modules.safetensors'}`",
        f"- `{out_dir / 'metrics.json'}`",
        f"- `{out_dir / 'teacher_labels.pt'}`",
    ])
    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/wall_x/pretrained/wall-oss-fast")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/wall_x/datasheet/libero_all")
    parser.add_argument("--repo-id", default="libero_all")
    parser.add_argument("--image-key", default="observation.images.faceImg")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_1000train_200val_step4")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--sample-strategy", choices=["linspace", "random"], default="linspace")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-pruning", action="store_true", default=False)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--pruned-strategy", default="topk_attention")
    parser.add_argument("--predictor-checkpoint", default=None)
    parser.add_argument("--predictor-source", default="early_hidden")
    parser.add_argument("--predictor-early-layer", type=int, default=None)
    parser.add_argument("--teacher-num-inference-timesteps", type=int, default=10)
    parser.add_argument("--student-num-inference-timesteps", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--teacher-seed", type=int, default=100000)
    parser.add_argument("--student-seed", type=int, default=200000)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--eval-timing-samples", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--regenerate-teacher", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.ode_distill = get_ode_distill_config(
        {
            "ode_distill": {
                "enable": True,
                "student_checkpoint_path": str(out_dir),
                "teacher_num_inference_timesteps": args.teacher_num_inference_timesteps,
                "student_num_inference_timesteps": args.student_num_inference_timesteps,
                "lora": {
                    "enable": True,
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": args.lora_target_modules,
                },
            }
        }
    )
    distill_config = {
        "ode_distill": args.ode_distill,
        "data": {
            "dataset_root": args.dataset_root,
            "repo_id": args.repo_id,
            "image_key": args.image_key,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "sample_strategy": args.sample_strategy,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "model_path": args.model_path,
        "vispruner": {
            "enable": args.enable_pruning,
            "keep_ratio": args.keep_ratio,
            "strategy": args.pruned_strategy,
        },
    }

    total_count = args.train_samples + args.val_samples
    items = make_items(args, total_count)
    teacher_payload, teacher_meta = build_teacher_cache(args, out_dir, items)
    teacher_actions = teacher_payload["teacher_actions"]

    student = prepare_student(args)
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_indices = list(range(args.train_samples))
    val_indices = list(range(args.train_samples, total_count))
    global_step = 0
    loss_history = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(train_indices)
        epoch_losses = []
        for pos, item_idx in enumerate(train_indices, start=1):
            item = items[item_idx]
            batch = make_image_batch(student, item["sample"], item["image_key"], args)
            target = teacher_actions[item_idx].to(args.device).unsqueeze(0)
            set_seed(args.student_seed + item["local_index"], args.device)
            out = call_flow_with_grad(
                student, batch, args, args.student_num_inference_timesteps
            )
            pred = out["predict_action"].to(torch.float32)
            loss = masked_smooth_l1(pred, target.to(torch.float32), batch["dof_mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in student.parameters() if param.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu().item()))
            if pos % args.log_every == 0 or pos == len(train_indices):
                print(
                    f"[ODE_DISTILL][train] epoch={epoch}/{args.epochs} "
                    f"step={pos}/{len(train_indices)} loss={np.mean(epoch_losses[-args.log_every:]):.6f}",
                    flush=True,
                )
        loss_history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses))})
        print(f"[ODE_DISTILL] epoch {epoch} avg_loss={loss_history[-1]['loss']:.6f}", flush=True)

    train_eval_indices = train_indices[: min(args.val_samples, len(train_indices))]
    train_eval = eval_student(args, student, items, teacher_actions, train_eval_indices, "train_eval")
    val_eval = eval_student(args, student, items, teacher_actions, val_indices, "val")
    metrics = {
        "epoch": args.epochs,
        "global_step": global_step,
        "elapsed_train_sec": time.perf_counter() - train_start,
        "loss_history": loss_history,
        "teacher_meta": teacher_meta,
        "train_eval": train_eval,
        "val": val_eval,
    }

    save_checkpoint(args, out_dir, student, metrics, distill_config)
    write_eval_report(args, out_dir, metrics)
    print(f"[ODE_DISTILL] saved to {out_dir}", flush=True)
    print(f"[ODE_DISTILL] val_mae={val_eval['mae']:.6f} val_rmse={val_eval['rmse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
