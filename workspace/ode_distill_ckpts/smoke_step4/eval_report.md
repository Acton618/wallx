# Wall-X ODE Distillation Report

- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- train_samples: `2`
- val_samples: `1`
- teacher_num_inference_timesteps: `10`
- student_num_inference_timesteps: `4`
- epochs: `1`
- learning_rate: `1e-05`
- vispruner_enable: `False`

## Summary

- train_eval MAE: `1.132654`
- val MAE: `1.159722`
- val RMSE: `1.460344`
- val max_abs: `4.869729`
- val endpoint_mae: `1.033944`

## Student Timing Samples

| stage | ms |
|---|---:|
| `total_time` | 186.707 |
| `prefetch_forward` | 36.133 |
| `ode_integration` | 117.350 |
| `ode_transformer_total` | 114.733 |
| `postprocessing` | 0.009 |

## Output Files

- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/smoke_step4/distill_config.yml`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/smoke_step4/adapter_model.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/smoke_step4/action_modules.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/smoke_step4/metrics.json`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/smoke_step4/teacher_labels.pt`