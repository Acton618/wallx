# Wall-X ODE Distillation Report

- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- train_samples: `5000`
- val_samples: `1000`
- teacher_num_inference_timesteps: `10`
- student_num_inference_timesteps: `4`
- epochs: `3`
- learning_rate: `2e-05`
- vispruner_enable: `False`

## Summary

- train_eval MAE: `0.801976`
- val MAE: `0.802183`
- val RMSE: `1.004558`
- val max_abs: `3.322770`
- val endpoint_mae: `0.803916`

## Student Timing Samples

| stage | ms |
|---|---:|
| `total_time` | 214.294 |
| `prefetch_forward` | 43.491 |
| `ode_integration` | 138.333 |
| `ode_transformer_total` | 135.386 |
| `postprocessing` | 0.009 |

## Output Files

- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4_lr2e-5_qkvo/distill_config.yml`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4_lr2e-5_qkvo/adapter_model.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4_lr2e-5_qkvo/action_modules.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4_lr2e-5_qkvo/metrics.json`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4_lr2e-5_qkvo/teacher_labels.pt`