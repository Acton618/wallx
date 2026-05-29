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
- learning_rate: `1e-05`
- vispruner_enable: `False`

## Summary

- train_eval MAE: `0.802261`
- val MAE: `0.802375`
- val RMSE: `1.004836`
- val max_abs: `3.321169`
- val endpoint_mae: `0.803075`

## Student Timing Samples

| stage | ms |
|---|---:|
| `total_time` | 186.664 |
| `prefetch_forward` | 36.485 |
| `ode_integration` | 118.857 |
| `ode_transformer_total` | 116.452 |
| `postprocessing` | 0.009 |

## Output Files

- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4/distill_config.yml`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4/adapter_model.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4/action_modules.safetensors`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4/metrics.json`
- `/root/autodl-tmp/wall_x/workspace/ode_distill_ckpts/ode_student_5000train_1000val_step4/teacher_labels.pt`