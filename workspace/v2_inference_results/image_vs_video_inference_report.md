# V2 Image vs Video Inference Result

- device: `cuda`
- prompt: `pick up the object`
- seed: `1234`
- ODE steps for display: `3`
- unnorm: `False`

## Image Input
- frame: `59` from each camera
- action shape: `[1, 32, 20]`
- image tokens/features: `162` / `162`
- image_grid_thw: `[[1, 18, 18], [1, 18, 18]]`
- first step first 7 dims: `[-2.537507, -0.855944, 4.789412, 1.646224, -1.80792, -0.47477, 3.327271]`
- elapsed_ms: `448.326`

## Video Input
- clip frames: `[56, 57, 58, 59]` from each camera
- action shape: `[1, 32, 20]`
- video tokens/features: `324` / `324`
- video_grid_thw: `[[2, 18, 18], [2, 18, 18]]`
- second_per_grid_ts: `[0.06666667014360428, 0.06666667014360428]`
- first step first 7 dims: `[-2.533498, -0.880306, 4.766153, 1.607757, -1.781473, -0.499249, 3.275504]`
- elapsed_ms: `205.156`

## Comparison
- first_3_steps_first_7_mean_abs_diff: `0.027409`
- first_3_steps_first_7_max_abs_diff: `0.067147`
- image_tokens_match_features: `True`
- video_tokens_match_features: `True`

## Preview Files
- `image_input_face_view_frame059.png`
- `image_input_right_wrist_view_frame059.png`
- `video_input_4frames_side_by_side.mp4`

Note: `unnorm=False` is used so this result focuses on pipeline comparison rather than robot-executable calibrated action values.