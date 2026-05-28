# Wall-X ODE Image Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `50`
- vispruner_enable: `True`
- keep_ratio: `0.5`
- num_inference_timesteps: `10`
- modified_mode: `early_stop`
- early_stop_patience: `1`
- early_stop_abs_threshold: `0.05`
- early_stop_rel_threshold: `0.1`
- warmup: `1`
- iters: `3`
- device: `cuda`

## Summary

- samples: `50`
- tokens: `81.00` -> `41.00`
- original total_time: `345.364 ms`
- modified total_time: `342.266 ms`
- total_time delta: `-3.098 ms` (`-0.90%`)
- original ode_integration: `285.671 ms`
- modified ode_integration: `283.020 ms`
- ode_integration delta: `-2.651 ms` (`-0.93%`)
- original ode_steps_used: `9.00`
- modified ode_steps_used: `9.00`
- action MAE vs original: `0.004703`
- action RMSE vs original: `0.005908`
- action max_abs vs original: `0.019801`

## Stage Timing

| stage | original_ms | modified_ms | delta_ms | delta_pct |
|---|---:|---:|---:|---:|
| `total_time` | 345.364 | 342.266 | -3.098 | -0.90% |
| `external_prepare_batch_ms` | 4.010 | 4.044 | +0.034 | +0.85% |
| `embed_processing` | 27.268 | 27.025 | -0.243 | -0.89% |
| `image_path_total` | 26.869 | 26.627 | -0.243 | -0.90% |
| `vision_image_forward` | 26.897 | 26.655 | -0.243 | -0.90% |
| `vision_image_encode_score` | 25.254 | 25.014 | -0.240 | -0.95% |
| `scatter_image_embeds` | 0.108 | 0.108 | -0.000 | -0.29% |
| `position_encoding` | 0.115 | 0.114 | -0.000 | -0.24% |
| `action_initialization` | 0.448 | 0.447 | -0.002 | -0.38% |
| `prefetch_forward` | 30.447 | 30.253 | -0.194 | -0.64% |
| `prefill_transformer` | 30.241 | 30.049 | -0.192 | -0.64% |
| `cache_preprocessing` | 1.278 | 1.271 | -0.007 | -0.54% |
| `ode_integration` | 285.671 | 283.020 | -2.651 | -0.93% |
| `ode_action_embed_total` | 2.753 | 2.733 | -0.020 | -0.74% |
| `ode_prepare_inputs` | 0.653 | 0.654 | +0.001 | +0.15% |
| `ode_transformer_total` | 279.063 | 276.768 | -2.296 | -0.82% |
| `ode_action_head_total` | 0.998 | 0.993 | -0.005 | -0.50% |
| `postprocessing` | 0.006 | 0.006 | +0.000 | +0.58% |
| `action_init_embed` | 0.268 | 0.266 | -0.002 | -0.56% |
| `action_init_noise` | 0.049 | 0.049 | +0.001 | +1.41% |
| `attention_mask_to_device` | 0.006 | 0.006 | +0.000 | +1.27% |
| `embed_tokens` | 0.045 | 0.045 | -0.000 | -0.35% |
| `image_cast` | 0.051 | 0.050 | -0.001 | -2.88% |
| `kv_cache_trim` | 0.819 | 0.817 | -0.002 | -0.21% |
| `moe_indices` | 0.093 | 0.093 | -0.000 | -0.25% |
| `postfix_mask_build` | 0.134 | 0.133 | -0.002 | -1.30% |
| `postfix_moe_indices` | 0.107 | 0.105 | -0.001 | -1.37% |
| `postfix_slice` | 0.056 | 0.055 | -0.000 | -0.80% |
| `prefill_action_head` | 0.150 | 0.149 | -0.001 | -0.52% |
| `prefix_length_resolve` | 0.077 | 0.076 | -0.000 | -0.61% |
| `pruning_position_ids_prepare` | 0.569 | 0.568 | -0.001 | -0.13% |
| `scatter_action_init` | 0.058 | 0.057 | -0.001 | -0.90% |
| `scatter_proprioception` | 0.126 | 0.127 | +0.000 | +0.28% |
| `vispruner_apply_keep_to_sequences` | 0.197 | 0.196 | -0.001 | -0.41% |
| `vispruner_build_keep_mask` | 0.242 | 0.243 | +0.001 | +0.31% |
| `vispruner_gather_image_embeds` | 0.041 | 0.041 | +0.000 | +1.02% |
| `vispruner_image_lengths` | 0.064 | 0.065 | +0.000 | +0.18% |
| `vispruner_pad_pruned_batch` | 0.079 | 0.078 | -0.001 | -1.40% |
| `vispruner_rope_deltas` | 0.132 | 0.132 | -0.000 | -0.26% |
| `vispruner_score_prepare` | 0.032 | 0.031 | -0.000 | -0.93% |
| `vispruner_topk_select` | 0.135 | 0.136 | +0.001 | +0.65% |
| `vispruner_total` | 0.859 | 0.858 | -0.000 | -0.04% |

## Paired Samples

| idx | source | tokens | original_steps | modified_steps | original_total_ms | modified_total_ms | total_delta_pct | original_ode_ms | modified_ode_ms | ode_delta_pct | action_mae | action_rmse | action_max_abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.417 | 343.253 | -0.34% | 285.672 | 284.172 | -0.53% | 0.004736 | 0.005900 | 0.017990 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.075 | 345.249 | -0.24% | 286.134 | 285.893 | -0.08% | 0.004836 | 0.006063 | 0.020499 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.081 | 347.056 | -0.01% | 287.496 | 287.432 | -0.02% | 0.004751 | 0.005911 | 0.020555 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.356 | 343.809 | -0.45% | 286.181 | 285.127 | -0.37% | 0.004454 | 0.005615 | 0.018546 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.366 | 343.215 | -0.91% | 287.255 | 284.121 | -1.09% | 0.004773 | 0.006064 | 0.021203 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.429 | 344.623 | +0.06% | 283.825 | 284.898 | +0.38% | 0.004853 | 0.006067 | 0.020969 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.897 | 343.666 | -0.93% | 287.225 | 285.192 | -0.71% | 0.004924 | 0.006155 | 0.022814 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.793 | 342.742 | -1.17% | 288.076 | 284.512 | -1.24% | 0.004770 | 0.005943 | 0.018163 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.060 | 342.190 | -1.12% | 285.895 | 282.904 | -1.05% | 0.004551 | 0.005731 | 0.017377 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.913 | 338.364 | -2.46% | 286.832 | 280.096 | -2.35% | 0.004417 | 0.005574 | 0.017667 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 342.576 | 339.883 | -0.79% | 283.879 | 281.762 | -0.75% | 0.004712 | 0.005942 | 0.018536 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.555 | 340.407 | -1.20% | 285.628 | 282.154 | -1.22% | 0.004531 | 0.005724 | 0.020483 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 343.394 | 339.542 | -1.12% | 283.925 | 280.965 | -1.04% | 0.004914 | 0.006110 | 0.019316 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.225 | 339.348 | -1.42% | 284.883 | 281.061 | -1.34% | 0.004588 | 0.005766 | 0.018539 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.088 | 339.335 | -1.95% | 286.985 | 281.225 | -2.01% | 0.004709 | 0.005959 | 0.020882 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.639 | 340.343 | -2.10% | 288.148 | 281.507 | -2.30% | 0.004838 | 0.006067 | 0.018862 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.871 | 339.019 | -1.98% | 285.706 | 280.555 | -1.80% | 0.004466 | 0.005537 | 0.016715 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 348.463 | 342.131 | -1.82% | 288.662 | 283.051 | -1.94% | 0.004246 | 0.005347 | 0.019282 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.741 | 344.746 | -0.58% | 287.248 | 284.506 | -0.95% | 0.004781 | 0.006084 | 0.019559 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.238 | 338.173 | -0.61% | 280.694 | 279.803 | -0.32% | 0.004732 | 0.005952 | 0.018605 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.899 | 341.861 | -1.74% | 287.268 | 282.532 | -1.65% | 0.004518 | 0.005655 | 0.017456 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 364.467 | 339.064 | -6.97% | 297.679 | 280.974 | -5.61% | 0.004568 | 0.005697 | 0.017000 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 343.532 | 338.122 | -1.57% | 284.188 | 279.856 | -1.52% | 0.004711 | 0.005842 | 0.016713 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.199 | 340.853 | -1.83% | 287.030 | 282.468 | -1.59% | 0.004733 | 0.006020 | 0.020338 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.407 | 340.192 | -1.51% | 286.536 | 281.624 | -1.71% | 0.004527 | 0.005823 | 0.018737 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.601 | 342.630 | -1.43% | 288.005 | 283.994 | -1.39% | 0.004606 | 0.005740 | 0.020181 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 343.188 | 352.257 | +2.64% | 284.635 | 287.554 | +1.03% | 0.004507 | 0.005736 | 0.019446 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.249 | 340.717 | -1.88% | 287.757 | 282.271 | -1.91% | 0.004804 | 0.005999 | 0.022059 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.756 | 335.508 | -1.25% | 281.013 | 276.761 | -1.51% | 0.004759 | 0.006003 | 0.021463 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.483 | 335.791 | -1.09% | 279.481 | 276.583 | -1.04% | 0.004910 | 0.006163 | 0.018435 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.024 | 339.914 | -0.33% | 281.631 | 279.982 | -0.59% | 0.004966 | 0.006161 | 0.022416 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 339.898 | 338.897 | -0.29% | 280.658 | 279.258 | -0.50% | 0.004771 | 0.005914 | 0.021114 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.236 | 343.407 | -1.10% | 288.092 | 284.176 | -1.36% | 0.004490 | 0.005582 | 0.019478 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.918 | 341.624 | -0.95% | 284.031 | 282.826 | -0.42% | 0.004748 | 0.006062 | 0.026895 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.832 | 342.133 | +0.38% | 282.242 | 282.912 | +0.24% | 0.004881 | 0.006072 | 0.018281 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.300 | 340.582 | -1.37% | 285.861 | 281.806 | -1.42% | 0.004748 | 0.005990 | 0.021463 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.983 | 341.279 | +0.09% | 282.382 | 282.402 | +0.01% | 0.004495 | 0.005622 | 0.020296 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.187 | 341.023 | -1.21% | 286.592 | 282.521 | -1.42% | 0.004851 | 0.006023 | 0.019043 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 343.190 | 341.250 | -0.57% | 284.600 | 282.686 | -0.67% | 0.004783 | 0.005956 | 0.016944 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.714 | 340.846 | +0.04% | 282.208 | 282.220 | +0.00% | 0.004659 | 0.005802 | 0.018767 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 344.955 | 340.948 | -1.16% | 286.301 | 282.285 | -1.40% | 0.004749 | 0.005922 | 0.020370 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 341.655 | 342.474 | +0.24% | 283.102 | 283.750 | +0.23% | 0.004489 | 0.005717 | 0.016250 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 346.798 | 343.293 | -1.01% | 286.497 | 284.274 | -0.78% | 0.005108 | 0.006342 | 0.022980 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 359.382 | 344.952 | -4.02% | 292.798 | 285.567 | -2.47% | 0.004748 | 0.006081 | 0.025501 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.364 | 344.687 | -0.20% | 285.643 | 285.507 | -0.05% | 0.004818 | 0.006000 | 0.018897 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.435 | 337.577 | -0.84% | 280.649 | 277.966 | -0.96% | 0.004672 | 0.005817 | 0.018787 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 340.939 | 381.954 | +12.03% | 282.529 | 308.217 | +9.09% | 0.004911 | 0.006270 | 0.024894 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 347.417 | 341.811 | -1.61% | 287.784 | 283.477 | -1.50% | 0.004585 | 0.005879 | 0.018686 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 350.929 | 341.148 | -2.79% | 289.992 | 282.201 | -2.69% | 0.004585 | 0.005877 | 0.021232 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 81->41 | 9.00 | 9.00 | 345.065 | 339.413 | -1.64% | 286.018 | 281.393 | -1.62% | 0.004862 | 0.006121 | 0.019379 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_early_stop_abs0.05_rel0.10_results.json`