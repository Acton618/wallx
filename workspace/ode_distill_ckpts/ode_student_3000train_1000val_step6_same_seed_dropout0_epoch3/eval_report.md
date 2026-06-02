# Wall-X ODE Distillation Report

- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- train_samples: `3000`
- val_samples: `1000`
- teacher_num_inference_timesteps: `10`
- student_num_inference_timesteps: `6`
- epochs: `3`
- learning_rate: `1e-05`
- vispruner_enable: `False`

## Summary

- train_eval MAE: `0.025816`
- val MAE: `0.025913`
- val RMSE: `0.032353`
- val max_abs: `0.103658`
- val endpoint_mae: `0.027687`

## Student Timing Samples

| stage | ms |
|---|---:|
| `total_time` | 246.948 |
| `prefetch_forward` | 34.007 |
| `ode_integration` | 183.776 |
| `ode_transformer_total` | 180.188 |
| `postprocessing` | 0.008 |

## Output Files

- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_3000train_1000val_step6_same_seed_dropout0_epoch3/distill_config.yml`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_3000train_1000val_step6_same_seed_dropout0_epoch3/adapter_model.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_3000train_1000val_step6_same_seed_dropout0_epoch3/action_modules.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_3000train_1000val_step6_same_seed_dropout0_epoch3/metrics.json`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_3000train_1000val_step6_same_seed_dropout0_epoch3/teacher_labels.pt`