# Wall-X V6 First-Pass ODE Distillation Implementation Report

## Goal

V6 adds a standalone ODE-distilled student path. The first version intentionally does not combine with V5 ODE velocity cache or V3 ODE early stop.

## Code Changes

- `wall_x/model/ode_distill_utils.py`
  - Added `allow_ode_cache` and `allow_ode_early_stop` to `ode_distill` config.
  - When loading a student checkpoint, V5 cache and V3 early stop are disabled unless explicitly allowed.

- `wall_x/model/qwen2_5_based/configuration_qwen2_5_vl.py`
  - Added persisted config fields for V6 distillation and combination flags.

- `wall_x/model/model_utils.py`
  - Reads V6 distillation flags from YAML.
  - If `ode_distill.enable=true`, disables `ode_cache` and `ode_early_stop` by default.

- `scripts/train_ode_distill_lerobot.py`
  - Default dataset input is `/root/autodl-tmp/wall_x/datasheet`.
  - Automatically resolves to `/root/autodl-tmp/wall_x/datasheet/libero_all` when `repo_id=libero_all`.
  - Default student steps changed to 6 for the first stable V6 path.
  - Runtime calls explicitly pass `ode_cache_enable=False` and `ode_early_stop_enable=False` unless combination flags are used.
  - Reports now record resolved dataset path and standalone V6 flags.

- `scripts/infer_robochallenge.py`
  - When a V6 student is enabled, uses `student_num_inference_timesteps`.
  - Normal flow path explicitly disables V5/V3 unless the student checkpoint/config allows them.

- `workspace/lerobot_example/config_qact_from_vlm.yml`
  - Added V6 combination flags under `ode_distill`.
  - Defaults remain disabled and standalone.

- `workspace/v6_ode_distill/enable_v6_student_example.yml`
  - Added a ready-to-use V6 student overlay example.

## Smoke Test

Command:

```bash
python3 scripts/train_ode_distill_lerobot.py   --dataset-root /root/autodl-tmp/wall_x/datasheet   --output-dir /root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6   --train-samples 1   --val-samples 1   --epochs 0   --eval-timing-samples 1   --log-every 1   --student-num-inference-timesteps 6   --teacher-num-inference-timesteps 10   --device cuda   --regenerate-teacher
```

Result:

- Dataset input: `/root/autodl-tmp/wall_x/datasheet`
- Dataset resolved: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- Teacher path: 10 ODE steps
- Student path: 6 ODE steps
- V6 standalone: `true`
- V5 cache enabled: `false`
- V3 early stop enabled: `false`
- Smoke report: `/root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6/eval_report.md`

Note: this smoke uses `epochs=0`, so the MAE only proves the code path runs; it is not a trained student quality number.

## Next Training Command

A real first training run can start with:

```bash
python3 scripts/train_ode_distill_lerobot.py   --dataset-root /root/autodl-tmp/wall_x/datasheet   --output-dir /root/autodl-tmp/wall_x/workspace/v6_ode_distill/ode_student_3000train_1000val_step6   --train-samples 3000   --val-samples 1000   --epochs 3   --student-num-inference-timesteps 6   --teacher-num-inference-timesteps 10   --device cuda   --log-every 25
```
