# Wall-X ODE Distillation Report

- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- dataset_root_input: `/root/autodl-tmp/wall_x/datasheet`
- dataset_root_resolved: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- train_samples: `1`
- val_samples: `1`
- teacher_num_inference_timesteps: `10`
- student_num_inference_timesteps: `6`
- epochs: `0`
- learning_rate: `1e-05`
- vispruner_enable: `False`
- V6 standalone: `True`
- allow_ode_cache_with_distill: `False`
- allow_ode_early_stop_with_distill: `False`

## Summary

- train_eval MAE: `1.097245`
- val MAE: `1.143421`
- val RMSE: `1.406808`
- val max_abs: `4.431957`
- val endpoint_mae: `0.950722`

## Student Timing Samples

| stage | ms |
|---|---:|
| `total_time` | 256.235 |
| `prefetch_forward` | 35.253 |
| `ode_integration` | 190.387 |
| `ode_transformer_total` | 187.120 |
| `postprocessing` | 0.006 |

## Output Files

- `/root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6/distill_config.yml`
- `/root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6/adapter_model.safetensors`
- `/root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6/action_modules.safetensors`
- `/root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6/metrics.json`
- `/root/autodl-tmp/wall_x/workspace/v6_ode_distill/smoke_step6/teacher_labels.pt`