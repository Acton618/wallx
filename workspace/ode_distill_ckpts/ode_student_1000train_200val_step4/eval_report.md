# Wall-X ODE Distillation Report

- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- train_samples: `1000`
- val_samples: `200`
- teacher_num_inference_timesteps: `10`
- student_num_inference_timesteps: `4`
- epochs: `3`
- learning_rate: `1e-05`
- vispruner_enable: `False`

## Summary

- train_eval MAE: `0.804754`
- val MAE: `0.806288`
- val RMSE: `1.008914`
- val max_abs: `3.351020`
- val endpoint_mae: `0.812206`

## Student Timing Samples

| stage | ms |
|---|---:|
| `total_time` | 182.266 |
| `prefetch_forward` | 36.728 |
| `ode_integration` | 115.517 |
| `ode_transformer_total` | 112.974 |
| `postprocessing` | 0.008 |

## Output Files

- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_1000train_200val_step4/distill_config.yml`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_1000train_200val_step4/adapter_model.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_1000train_200val_step4/action_modules.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_1000train_200val_step4/metrics.json`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_1000train_200val_step4/teacher_labels.pt`