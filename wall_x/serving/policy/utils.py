from typing import Dict, List
import logging
import numpy as np
from wall_x.data.utils import preprocesser_call
from qwen_vl_utils.vision_process import smart_resize
import torch
from PIL import Image
from transformers import BatchFeature

logger = logging.getLogger(__name__)


def _to_pil_image(img) -> Image.Image:
    """Convert a serving image/frame to PIL while preserving the old image path."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim > 3:
            img = img.squeeze()
        if img.ndim == 3 and img.shape[0] in {1, 3, 4}:
            img = img.permute(1, 2, 0)
        img = img.numpy()
    if isinstance(img, np.ndarray):
        if img.ndim > 3:
            logger.warning(
                f"Image/frame has {img.ndim} dimensions, squeezing extra dimensions"
            )
            img = np.squeeze(img)
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            if img.shape[0] in {1, 3, 4}:
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[2] in {1, 3, 4}:
                pass
            else:
                raise ValueError(
                    f"Unexpected image/frame shape: {img.shape}. "
                    "Expected (H, W, C) or (C, H, W)."
                )
        else:
            raise ValueError(
                f"Invalid image/frame dimensions: {img.ndim}; got shape {img.shape}"
            )
        if img.dtype == np.uint8:
            return Image.fromarray(img).convert("RGB")
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert(
            "RGB"
        )
    raise TypeError(f"Unsupported image/frame type: {type(img)!r}")


def _prepare_image_inputs(
    obs: Dict, camera_key: List[str], image_factor, min_pixels, max_pixels
):
    images = [_to_pil_image(obs[key]) for key in camera_key]
    return process_images(images, image_factor, min_pixels, max_pixels)


def _prepare_video_inputs(
    obs: Dict, camera_key: List[str], image_factor, min_pixels, max_pixels
):
    """V2 video path: build one ordered clip per camera placeholder.

    Expected schema:
        obs["media_type"] = "video"
        obs["video_frames"][camera_name] = [frame0, frame1, ...]
    """
    video_frames = obs.get("video_frames", None)
    if video_frames is None:
        raise ValueError(
            "media_type='video' requires obs['video_frames'][camera_key] = frames."
        )

    clips = []
    for cam_name in camera_key:
        if cam_name not in video_frames:
            raise KeyError(f"Missing video frames for camera {cam_name!r}.")
        frames = video_frames[cam_name]
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            frame_list = [frames[i] for i in range(frames.shape[0])]
        elif isinstance(frames, torch.Tensor) and frames.ndim == 4:
            frame_list = [frames[i] for i in range(frames.shape[0])]
        else:
            frame_list = list(frames)
        if not frame_list:
            raise ValueError(f"Camera {cam_name!r} has no video frames.")

        pil_frames = [_to_pil_image(frame) for frame in frame_list]
        resized_frames = process_images(
            pil_frames, image_factor, min_pixels, max_pixels
        )
        # Qwen video processors accept a clip as a list of RGB arrays/PIL frames.
        clips.append([np.array(frame) for frame in resized_frames])
    return clips


def _resolve_second_per_grid_ts(obs: Dict, inputs):
    """V2 video path: optionally make temporal RoPE scale explicit."""
    if "video_grid_thw" not in inputs or inputs["video_grid_thw"] is None:
        return None
    value = obs.get("second_per_grid_ts", None)
    if value is None:
        value = obs.get("video_seconds_per_grid", None)
    if value is None and obs.get("video_fps", None):
        value = 1.0 / float(obs["video_fps"])
    if value is None:
        return None

    video_count = inputs["video_grid_thw"].shape[0]
    if torch.is_tensor(value):
        tensor = value.to(dtype=torch.float32)
    elif isinstance(value, (list, tuple, np.ndarray)):
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        tensor = torch.full((video_count,), float(value), dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.repeat(video_count)
    if tensor.numel() == 1 and video_count > 1:
        tensor = tensor.repeat(video_count)
    if tensor.numel() != video_count:
        raise ValueError(
            f"second_per_grid_ts has {tensor.numel()} values, expected {video_count}."
        )
    return tensor


def prepare_batch(
    obs: Dict,
    processor,
    normalizer_propri,
    camera_key: List[str],
    agent_pos_dim,
    action_dim,
    pred_horizon,
    fixed_action_dim,
    max_length,
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    predict_mode: str = "fast",
    device: str = "cuda",
) -> BatchFeature:
    """Prepare observation into model input format.

    Args:
        obs: Dictionary containing:
            - image mode: obs[camera_name] = image
            - video mode: obs["media_type"] = "video" and
              obs["video_frames"][camera_name] = [frame0, frame1, ...]
            - 'prompt': Text prompt
            - 'state': Robot state/proprioception
            - 'dataset_names': Dataset names

    Returns:
        BatchFeature object ready for model input
    """
    media_type = obs.get("media_type", "image")
    if media_type not in {"image", "video"}:
        raise ValueError(
            f"Unsupported media_type={media_type!r}; expected image or video."
        )

    # Handle visual observations. Image mode preserves the original serving schema;
    # V2 video mode consumes obs["video_frames"][camera_name] clips.
    if media_type == "video":
        image_inputs = None
        video_inputs = _prepare_video_inputs(
            obs, camera_key, image_factor, min_pixels, max_pixels
        )
    else:
        image_inputs = [
            _prepare_image_inputs(obs, camera_key, image_factor, min_pixels, max_pixels)
        ]
        video_inputs = None

    # Handle text prompt - format with matching image/video vision tokens.
    instruction = obs["prompt"]
    formatted_text = format_text_with_vision_tokens(
        instruction, camera_key, predict_mode, pred_horizon, media_type=media_type
    )

    # Use processor to prepare inputs
    inputs = preprocesser_call(
        processor=processor,
        text=[formatted_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    second_per_grid_ts = _resolve_second_per_grid_ts(obs, inputs)
    if second_per_grid_ts is not None and "second_per_grid_ts" not in inputs:
        # V2 video path: keep temporal RoPE metadata on the inference device.
        inputs["second_per_grid_ts"] = second_per_grid_ts.to(device)

    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    inputs["moe_token_types"] = inputs.input_ids == action_token_id

    # obs["dataset_names"]="libero_all"

    # Handle robot state/proprioception if available
    if "state" in obs:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        # Pad to 20 dimensions if needed (same as training)
        # if state.shape[-1] < 20:
        #     padding = torch.zeros(state.shape[0], state.shape[1], 20 - state.shape[-1])
        #     state = torch.cat([state, padding], dim=-1)

        # Create mask for valid dimensions
        agent_pos_mask = torch.ones_like(state)
        if state.shape[-1] > agent_pos_dim:
            agent_pos_mask[:, :, agent_pos_dim:] = 0

        normalizer_propri.normalize_data(
            state, [obs["dataset_names"]] * state.shape[0]
        )

        inputs["proprioception"] = state
        inputs["agent_pos_mask"] = agent_pos_mask

    # Add dataset name (required by model)
    inputs["dataset_names"] = [obs["dataset_names"]] * state.shape[0]

    # Move all tensors to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    dof_mask = torch.ones([state.shape[0], pred_horizon, fixed_action_dim])
    dof_mask[:, :, action_dim:] = 0

    inputs["dof_mask"] = dof_mask

    # Convert to BatchFeature to maintain consistency with training pipeline
    return BatchFeature(data=dict(inputs)).to(device)


def process_images(
    images: List[Image.Image], image_factor: int, min_pixels: int, max_pixels: int
) -> List[Image.Image]:
    """Process images with smart resize following the data loading pattern.

    Args:
        images: List of PIL Images

    Returns:
        List of resized PIL Images
    """
    resized_images = []
    for img_pil in images:

        orig_width, orig_height = img_pil.size
        target_size = 256
        if target_size != -1:
            # Maintain aspect ratio logic
            if orig_width > orig_height:  # Landscape image
                new_width = target_size
                new_height = int(target_size * orig_height / orig_width)
            else:  # Portrait image
                new_height = target_size
                new_width = int(target_size * orig_width / orig_height)
            img_pil = img_pil.resize((new_width, new_height))

        # Apply smart scaling (Qwen logic)
        current_width, current_height = img_pil.size
        resized_height, resized_width = smart_resize(
            current_height,
            current_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        resized_img = img_pil.resize((resized_width, resized_height))
        resized_images.append(resized_img)

    return resized_images


def format_text_with_vision_tokens(
    instruction: str,
    camera_key: List[str],
    predict_mode: str = "diffusion",
    pred_horizon: int = 32,
    media_type: str = "image",
) -> str:
    """Format text prompt with vision tokens for the model.

    Args:
        instruction: Task instruction text
        camera_key: List of camera names

    Returns:
        Formatted text with special tokens
    """
    # Special tokens for formatting
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    if media_type not in {"image", "video"}:
        raise ValueError(
            f"Unsupported media_type={media_type!r}; expected image or video."
        )
    image_pad_symbol = "<|image_pad|>"
    video_pad_symbol = "<|video_pad|>"
    # V2 video path: prompt placeholder must match the selected media branch.
    vision_pad_symbol = video_pad_symbol if media_type == "video" else image_pad_symbol
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    action_fast_symbol = "<|action_fast|>"

    # Camera name mapping
    camera_name_mapping = {
        "front_view": "front view",
        "face_view": "front view",
        "left_wrist_view": "left wrist view",
        "right_wrist_view": "right wrist view",
        "top_view": "top view",
        "wall_view": "wall view",
    }

    # System prologue
    prologue = (
        f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    )

    # User request with observation
    user_request = f"{role_start_symbol}user\nObservation:"
    if camera_key:
        for cam_name in camera_key:
            view_name = camera_name_mapping.get(cam_name, cam_name)
            user_request += (
                f" {view_name}: "
                f"{vision_start_symbol}{vision_pad_symbol}{vision_end_symbol}"
            )
    user_request += "\nInstruction:"

    text_prompt = (
        f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
    )
    user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
    assistant_output = (
        f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n"
    )
    if predict_mode == "diffusion":
        assistant_output = f"{role_start_symbol}assistant\n{action_symbol * pred_horizon}{role_end_symbol}\n"
    complete_text = prologue + user_message + assistant_output

    return complete_text
