import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from safetensors.torch import load_file


def get_ode_distill_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config.get("ode_distill", {}) or {})
    cfg.setdefault("enable", False)
    cfg.setdefault("student_checkpoint_path", None)
    cfg.setdefault("teacher_num_inference_timesteps", 10)
    cfg.setdefault("student_num_inference_timesteps", 4)
    cfg.setdefault("lora", {})
    cfg["lora"] = dict(cfg.get("lora") or {})
    cfg["lora"].setdefault("enable", True)
    cfg["lora"].setdefault("r", 8)
    cfg["lora"].setdefault("alpha", 16)
    cfg["lora"].setdefault("dropout", 0.05)
    cfg["lora"].setdefault("target_modules", ["q_proj", "v_proj"])
    return cfg


def load_distill_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    config_path = Path(checkpoint_path) / "distill_config.yml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded


def _strip_action_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("action_preprocessor."):
            stripped[key[len("action_preprocessor.") :]] = value
        else:
            stripped[key] = value
    return stripped


def apply_ode_distill_checkpoint(model, config: Dict[str, Any], is_trainable: bool = False):
    cfg = get_ode_distill_config(config)
    if not cfg.get("enable", False):
        return model, cfg

    checkpoint_path = cfg.get("student_checkpoint_path")
    if not checkpoint_path:
        raise ValueError("ode_distill.enable is true but student_checkpoint_path is empty")

    checkpoint_path = str(checkpoint_path)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"ODE distill checkpoint not found: {checkpoint_path}")

    saved_cfg = load_distill_config_from_checkpoint(checkpoint_path)
    if saved_cfg:
        saved_ode_cfg = saved_cfg.get("ode_distill", saved_cfg)
        saved_lora_cfg = saved_ode_cfg.get("lora", {})
        merged_lora = dict(saved_lora_cfg or {})
        merged_lora.update(cfg.get("lora", {}) or {})
        saved_ode_cfg.update(cfg)
        saved_ode_cfg["lora"] = merged_lora
        cfg = get_ode_distill_config({"ode_distill": saved_ode_cfg})

    adapter_config = Path(checkpoint_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        model.model = PeftModel.from_pretrained(
            model.model, checkpoint_path, is_trainable=is_trainable
        )
    elif cfg["lora"].get("enable", True):
        model.add_lora(
            r=int(cfg["lora"].get("r", 8)),
            lora_alpha=int(cfg["lora"].get("alpha", 16)),
            target_modules=list(cfg["lora"].get("target_modules", ["q_proj", "v_proj"])),
            lora_dropout=float(cfg["lora"].get("dropout", 0.05)),
        )
        adapter_weights = Path(checkpoint_path) / "adapter_model.safetensors"
        if adapter_weights.exists():
            adapter_state = load_file(str(adapter_weights), device="cpu")
            model.model.load_state_dict(adapter_state, strict=False)

    action_path = Path(checkpoint_path) / "action_modules.safetensors"
    if action_path.exists():
        action_state = _strip_action_prefix(load_file(str(action_path), device="cpu"))
        missing, unexpected = model.action_preprocessor.load_state_dict(
            action_state, strict=False
        )
        if missing:
            print(f"[ODE_DISTILL] Missing action module keys: {missing}", flush=True)
        if unexpected:
            print(f"[ODE_DISTILL] Unexpected action module keys: {unexpected}", flush=True)

    model.ode_distill_enabled = True
    model.ode_distill_num_inference_timesteps = int(
        cfg.get("student_num_inference_timesteps", 4)
    )
    return model, cfg


def action_preprocessor_state_dict(model):
    state = {}
    for key, value in model.action_preprocessor.state_dict().items():
        if key.startswith("normalizer_"):
            continue
        state[f"action_preprocessor.{key}"] = value.detach().cpu()
    return state


def freeze_for_ode_distill(model):
    for param in model.parameters():
        param.requires_grad_(False)

    for param in model.action_preprocessor.parameters():
        param.requires_grad_(True)

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


def trainable_parameter_names(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]
