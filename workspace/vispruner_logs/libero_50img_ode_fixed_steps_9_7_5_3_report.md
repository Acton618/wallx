# Wall-X Fixed ODE Step Timing Report

- dataset_root: `/root/autodl-tmp/wall_x/datasheet/libero_all`
- repo_id: `libero_all`
- image_key: `observation.images.faceImg`
- model_path: `/root/autodl-tmp/wall_x/pretrained/wall-oss-fast`
- num_images: `50`
- vispruner_enable: `True`
- keep_ratio: `0.5`
- compared_steps: `9 / 7 / 5 / 3`
- warmup: `1`
- iters: `3`
- base_seed: `1234`
- device: `cuda`

## Summary

| ODE steps | total_ms | total_delta_pct_vs_9 | ode_ms | ode_delta_pct_vs_9 | action_mae_vs_9 | action_rmse_vs_9 | action_max_abs_vs_9 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 338.610 | +0.00% | 280.002 | +0.00% | 0.000000 | 0.000000 | 0.000000 |
| 7 | 273.841 | -19.13% | 216.027 | -22.85% | 0.007295 | 0.009179 | 0.032224 |
| 5 | 213.275 | -37.01% | 155.220 | -44.56% | 0.010493 | 0.013565 | 0.058227 |
| 3 | 151.333 | -55.31% | 93.309 | -66.68% | 0.018625 | 0.024823 | 0.121965 |

## Stage Timing

| stage | 9_step_ms | 7_step_ms | 5_step_ms | 3_step_ms |
|---|---:|---:|---:|---:|
| `total_time` | 338.610 | 273.841 | 213.275 | 151.333 |
| `external_prepare_batch_ms` | 3.527 | 3.282 | 3.235 | 3.181 |
| `embed_processing` | 26.860 | 26.427 | 26.501 | 26.480 |
| `image_path_total` | 26.457 | 26.043 | 26.120 | 26.097 |
| `vision_image_forward` | 26.485 | 26.070 | 26.148 | 26.125 |
| `position_encoding` | 0.116 | 0.109 | 0.109 | 0.109 |
| `action_initialization` | 0.442 | 0.431 | 0.431 | 0.432 |
| `prefetch_forward` | 29.783 | 29.464 | 29.633 | 29.623 |
| `prefill_transformer` | 29.581 | 29.266 | 29.435 | 29.424 |
| `cache_preprocessing` | 1.267 | 1.245 | 1.244 | 1.242 |
| `ode_integration` | 280.002 | 216.027 | 155.220 | 93.309 |
| `ode_transformer_total` | 273.551 | 211.047 | 151.603 | 91.055 |
| `postprocessing` | 0.009 | 0.009 | 0.009 | 0.009 |
| `action_init_embed` | 0.261 | 0.256 | 0.256 | 0.258 |
| `action_init_noise` | 0.049 | 0.047 | 0.045 | 0.046 |
| `attention_mask_to_device` | 0.007 | 0.006 | 0.006 | 0.006 |
| `embed_tokens` | 0.045 | 0.041 | 0.039 | 0.039 |
| `image_cast` | 0.049 | 0.045 | 0.044 | 0.044 |
| `kv_cache_trim` | 0.813 | 0.808 | 0.809 | 0.809 |
| `moe_indices` | 0.094 | 0.088 | 0.088 | 0.088 |
| `ode_action_embed_total` | 2.673 | 2.056 | 1.475 | 0.888 |
| `ode_action_head_total` | 0.970 | 0.742 | 0.532 | 0.322 |
| `ode_prepare_inputs` | 0.641 | 0.491 | 0.353 | 0.212 |
| `postfix_mask_build` | 0.133 | 0.126 | 0.123 | 0.124 |
| `postfix_moe_indices` | 0.104 | 0.101 | 0.103 | 0.100 |
| `postfix_slice` | 0.055 | 0.054 | 0.054 | 0.054 |
| `prefill_action_head` | 0.146 | 0.143 | 0.143 | 0.144 |
| `prefix_length_resolve` | 0.075 | 0.071 | 0.069 | 0.070 |
| `pruning_position_ids_prepare` | 0.552 | 0.521 | 0.514 | 0.513 |
| `scatter_action_init` | 0.057 | 0.056 | 0.056 | 0.056 |
| `scatter_image_embeds` | 0.106 | 0.101 | 0.100 | 0.101 |
| `scatter_proprioception` | 0.129 | 0.123 | 0.122 | 0.123 |
| `vision_image_encode_score` | 24.871 | 24.537 | 24.628 | 24.607 |
| `vispruner_apply_keep_to_sequences` | 0.196 | 0.190 | 0.191 | 0.190 |
| `vispruner_build_keep_mask` | 0.240 | 0.225 | 0.221 | 0.221 |
| `vispruner_gather_image_embeds` | 0.040 | 0.039 | 0.040 | 0.039 |
| `vispruner_image_lengths` | 0.063 | 0.060 | 0.058 | 0.058 |
| `vispruner_pad_pruned_batch` | 0.075 | 0.073 | 0.072 | 0.072 |
| `vispruner_rope_deltas` | 0.129 | 0.125 | 0.125 | 0.124 |
| `vispruner_score_prepare` | 0.030 | 0.028 | 0.028 | 0.028 |
| `vispruner_topk_select` | 0.133 | 0.125 | 0.122 | 0.122 |
| `vispruner_total` | 0.849 | 0.814 | 0.809 | 0.807 |

## Per-Sample Action Error

### 7 steps vs 9 steps

| idx | source | action_mae | action_rmse | action_max_abs |
|---:|---|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 0.007404 | 0.009180 | 0.030555 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 0.007243 | 0.009128 | 0.030330 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 0.007631 | 0.009473 | 0.028840 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 0.006939 | 0.008678 | 0.026166 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 0.007377 | 0.009283 | 0.032414 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 0.007262 | 0.009028 | 0.033448 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 0.007057 | 0.008693 | 0.024605 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 0.007706 | 0.009570 | 0.032484 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 0.007554 | 0.009509 | 0.033938 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 0.007320 | 0.009271 | 0.035433 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 0.007483 | 0.009308 | 0.028735 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 0.007190 | 0.008983 | 0.039593 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 0.007177 | 0.009048 | 0.028003 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 0.007041 | 0.008936 | 0.033361 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 0.007365 | 0.009407 | 0.031893 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 0.007034 | 0.008881 | 0.028252 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 0.007273 | 0.009029 | 0.033128 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 0.007382 | 0.009206 | 0.026274 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 0.007162 | 0.009050 | 0.033406 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 0.007256 | 0.009183 | 0.030290 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 0.007108 | 0.008941 | 0.029638 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 0.007325 | 0.009160 | 0.033165 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 0.007144 | 0.009129 | 0.034249 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 0.007498 | 0.009456 | 0.036191 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 0.007250 | 0.008948 | 0.034470 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 0.007270 | 0.009262 | 0.029472 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 0.006912 | 0.008621 | 0.024905 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 0.006846 | 0.008792 | 0.036183 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 0.007644 | 0.009836 | 0.044415 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 0.007781 | 0.009784 | 0.034434 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 0.007157 | 0.008977 | 0.026390 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 0.007184 | 0.009104 | 0.030996 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 0.007578 | 0.009429 | 0.030937 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 0.007497 | 0.009523 | 0.032148 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 0.007137 | 0.008897 | 0.030493 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 0.007451 | 0.009319 | 0.025555 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 0.007310 | 0.009112 | 0.033433 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 0.007078 | 0.008862 | 0.034092 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 0.007497 | 0.009435 | 0.028388 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 0.006971 | 0.008873 | 0.040028 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 0.007204 | 0.008876 | 0.026189 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 0.007225 | 0.009202 | 0.034356 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 0.007022 | 0.008827 | 0.032863 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 0.007306 | 0.009251 | 0.039071 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 0.007475 | 0.009467 | 0.035339 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 0.007518 | 0.009515 | 0.032281 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 0.007242 | 0.009187 | 0.036544 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 0.007505 | 0.009615 | 0.037822 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 0.007640 | 0.009638 | 0.034021 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 0.007140 | 0.009045 | 0.032005 |

### 5 steps vs 9 steps

| idx | source | action_mae | action_rmse | action_max_abs |
|---:|---|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 0.010283 | 0.013500 | 0.073075 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 0.010124 | 0.013203 | 0.053000 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 0.010930 | 0.014250 | 0.062773 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 0.010123 | 0.013129 | 0.052330 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 0.010371 | 0.013591 | 0.069201 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 0.010504 | 0.013919 | 0.057466 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 0.009943 | 0.012922 | 0.052676 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 0.010616 | 0.013472 | 0.047453 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 0.010415 | 0.013547 | 0.058770 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 0.010283 | 0.013277 | 0.064462 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 0.010671 | 0.014029 | 0.059374 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 0.010204 | 0.013163 | 0.059176 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 0.010197 | 0.013456 | 0.061156 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 0.009972 | 0.012973 | 0.064438 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 0.010597 | 0.013761 | 0.061622 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 0.010598 | 0.013417 | 0.050434 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 0.010163 | 0.012805 | 0.046736 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 0.010342 | 0.013069 | 0.049024 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 0.010716 | 0.013745 | 0.067864 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 0.011138 | 0.014311 | 0.062437 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 0.010489 | 0.013361 | 0.053368 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 0.010463 | 0.013464 | 0.054998 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 0.010828 | 0.014161 | 0.062841 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 0.010945 | 0.013914 | 0.067218 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 0.009773 | 0.012670 | 0.053728 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 0.010366 | 0.013361 | 0.054506 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 0.010024 | 0.012905 | 0.052193 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 0.010406 | 0.013530 | 0.055890 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 0.010977 | 0.014216 | 0.056990 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 0.010798 | 0.014116 | 0.074361 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 0.010160 | 0.013025 | 0.052167 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 0.010846 | 0.013833 | 0.055755 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 0.010379 | 0.013366 | 0.051891 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 0.010690 | 0.013698 | 0.056181 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 0.010476 | 0.013492 | 0.059971 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 0.009968 | 0.012859 | 0.055415 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 0.010382 | 0.013589 | 0.064390 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 0.010448 | 0.013572 | 0.059078 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 0.010936 | 0.014292 | 0.070369 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 0.010051 | 0.013179 | 0.051341 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 0.010243 | 0.013289 | 0.056937 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 0.010579 | 0.013744 | 0.061092 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 0.010895 | 0.013859 | 0.054691 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 0.010644 | 0.013644 | 0.052852 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 0.010813 | 0.013884 | 0.060967 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 0.010592 | 0.013684 | 0.053974 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 0.010971 | 0.014229 | 0.067854 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 0.011143 | 0.014385 | 0.062388 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 0.010927 | 0.014123 | 0.052919 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 0.010232 | 0.013254 | 0.051555 |

### 3 steps vs 9 steps

| idx | source | action_mae | action_rmse | action_max_abs |
|---:|---|---:|---:|---:|
| 1 | `dataset_index=0 image_key=observation.images.faceImg` | 0.018348 | 0.024221 | 0.110586 |
| 2 | `dataset_index=5546 image_key=observation.images.faceImg` | 0.018909 | 0.024709 | 0.100277 |
| 3 | `dataset_index=11092 image_key=observation.images.faceImg` | 0.019315 | 0.025424 | 0.107038 |
| 4 | `dataset_index=16639 image_key=observation.images.faceImg` | 0.018458 | 0.024744 | 0.111584 |
| 5 | `dataset_index=22185 image_key=observation.images.faceImg` | 0.018465 | 0.025518 | 0.149161 |
| 6 | `dataset_index=27731 image_key=observation.images.faceImg` | 0.018989 | 0.025377 | 0.123062 |
| 7 | `dataset_index=33278 image_key=observation.images.faceImg` | 0.017653 | 0.023373 | 0.103557 |
| 8 | `dataset_index=38824 image_key=observation.images.faceImg` | 0.019215 | 0.024954 | 0.112745 |
| 9 | `dataset_index=44370 image_key=observation.images.faceImg` | 0.018881 | 0.025645 | 0.134867 |
| 10 | `dataset_index=49917 image_key=observation.images.faceImg` | 0.018864 | 0.025452 | 0.130071 |
| 11 | `dataset_index=55463 image_key=observation.images.faceImg` | 0.018800 | 0.024606 | 0.117627 |
| 12 | `dataset_index=61009 image_key=observation.images.faceImg` | 0.018530 | 0.024493 | 0.123180 |
| 13 | `dataset_index=66556 image_key=observation.images.faceImg` | 0.017944 | 0.024217 | 0.128922 |
| 14 | `dataset_index=72102 image_key=observation.images.faceImg` | 0.018385 | 0.024208 | 0.109966 |
| 15 | `dataset_index=77648 image_key=observation.images.faceImg` | 0.019153 | 0.025031 | 0.103459 |
| 16 | `dataset_index=83195 image_key=observation.images.faceImg` | 0.018640 | 0.024827 | 0.134838 |
| 17 | `dataset_index=88741 image_key=observation.images.faceImg` | 0.018523 | 0.024273 | 0.119432 |
| 18 | `dataset_index=94287 image_key=observation.images.faceImg` | 0.018299 | 0.024211 | 0.107466 |
| 19 | `dataset_index=99834 image_key=observation.images.faceImg` | 0.017927 | 0.024107 | 0.117627 |
| 20 | `dataset_index=105380 image_key=observation.images.faceImg` | 0.018732 | 0.024695 | 0.122454 |
| 21 | `dataset_index=110926 image_key=observation.images.faceImg` | 0.019090 | 0.025065 | 0.119836 |
| 22 | `dataset_index=116473 image_key=observation.images.faceImg` | 0.017940 | 0.024430 | 0.122112 |
| 23 | `dataset_index=122019 image_key=observation.images.faceImg` | 0.019022 | 0.025180 | 0.129210 |
| 24 | `dataset_index=127565 image_key=observation.images.faceImg` | 0.018522 | 0.024822 | 0.128681 |
| 25 | `dataset_index=133112 image_key=observation.images.faceImg` | 0.018498 | 0.024598 | 0.129521 |
| 26 | `dataset_index=138658 image_key=observation.images.faceImg` | 0.018650 | 0.024394 | 0.130872 |
| 27 | `dataset_index=144205 image_key=observation.images.faceImg` | 0.017956 | 0.024026 | 0.132644 |
| 28 | `dataset_index=149751 image_key=observation.images.faceImg` | 0.017571 | 0.023860 | 0.111676 |
| 29 | `dataset_index=155297 image_key=observation.images.faceImg` | 0.018560 | 0.025148 | 0.118891 |
| 30 | `dataset_index=160844 image_key=observation.images.faceImg` | 0.018960 | 0.025199 | 0.136100 |
| 31 | `dataset_index=166390 image_key=observation.images.faceImg` | 0.018938 | 0.024766 | 0.112498 |
| 32 | `dataset_index=171936 image_key=observation.images.faceImg` | 0.018967 | 0.025002 | 0.134982 |
| 33 | `dataset_index=177483 image_key=observation.images.faceImg` | 0.018586 | 0.025239 | 0.127692 |
| 34 | `dataset_index=183029 image_key=observation.images.faceImg` | 0.018760 | 0.024892 | 0.121410 |
| 35 | `dataset_index=188575 image_key=observation.images.faceImg` | 0.018611 | 0.024919 | 0.123439 |
| 36 | `dataset_index=194122 image_key=observation.images.faceImg` | 0.018255 | 0.024498 | 0.124007 |
| 37 | `dataset_index=199668 image_key=observation.images.faceImg` | 0.018625 | 0.024464 | 0.112302 |
| 38 | `dataset_index=205214 image_key=observation.images.faceImg` | 0.018717 | 0.025336 | 0.123371 |
| 39 | `dataset_index=210761 image_key=observation.images.faceImg` | 0.019725 | 0.026427 | 0.142622 |
| 40 | `dataset_index=216307 image_key=observation.images.faceImg` | 0.017789 | 0.023995 | 0.105296 |
| 41 | `dataset_index=221853 image_key=observation.images.faceImg` | 0.018253 | 0.024216 | 0.119401 |
| 42 | `dataset_index=227400 image_key=observation.images.faceImg` | 0.018039 | 0.024343 | 0.132034 |
| 43 | `dataset_index=232946 image_key=observation.images.faceImg` | 0.018507 | 0.025070 | 0.130118 |
| 44 | `dataset_index=238492 image_key=observation.images.faceImg` | 0.018918 | 0.025262 | 0.112469 |
| 45 | `dataset_index=244039 image_key=observation.images.faceImg` | 0.018841 | 0.025257 | 0.133312 |
| 46 | `dataset_index=249585 image_key=observation.images.faceImg` | 0.019302 | 0.025640 | 0.117050 |
| 47 | `dataset_index=255131 image_key=observation.images.faceImg` | 0.018931 | 0.025830 | 0.144090 |
| 48 | `dataset_index=260678 image_key=observation.images.faceImg` | 0.019158 | 0.025284 | 0.115116 |
| 49 | `dataset_index=266224 image_key=observation.images.faceImg` | 0.018590 | 0.024564 | 0.114453 |
| 50 | `dataset_index=271771 image_key=observation.images.faceImg` | 0.018948 | 0.025330 | 0.125140 |

## Raw Results

- `/root/autodl-tmp/wall_x/workspace/vispruner_logs/libero_50img_ode_fixed_steps_9_7_5_3_results.json`