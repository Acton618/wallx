"""
LeRobot Dataset Loader - Distributed Version
"""

import numpy as np
import torch
from torch.utils.data import DistributedSampler, random_split
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from typing import Protocol, SupportsIndex, TypeVar
from qwen_vl_utils.vision_process import smart_resize
from wall_x.data.config import X2RDataProcessingConfig
from wall_x.data.utils import (
    process_grounding_points,
    get_wallx_normal_text,
    replace_action_token,
    preprocesser_call,
)

from transformers import AutoProcessor
from .utils import KEY_MAPPINGS

T_co = TypeVar("T_co", covariant=True)


# Abstract class for dataset
class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class PreprocessedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=42,
        rank=0,
        world_size=1,
        test_only=False,
    ):
        self.hf_dataset = dataset

        if test_only:
            self._dataset = dataset
        else:
            self._dataset = None
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [0.95, 0.05],
                torch.Generator().manual_seed(seed) if seed is not None else None,
            )
            self._train()

        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # init configs
        self.config = config
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.dataload_config = dataload_config
        self.normalizer_action = (normalizer_action,)
        self.normalizer_propri = normalizer_propri
        # self.norm_stats = norm_stats
        self.lerobot_config = lerobot_config

        self.data_config = X2RDataProcessingConfig().update(
            train_test_split=self.dataload_config["train_test_split"],
            split_seed=self.dataload_config["split_seed"],
            predict_action_keys=self.dataload_config["predict_action_keys"],
            obs_action_keys=self.dataload_config["obs_action_keys"],
            resolution=self.dataload_config.get("resolution", None),
            priority_order=self.dataload_config.get("priority_order", None),
        )

        self._cam_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]["camera"]
        self._state_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]
        self._action_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]
        self.media_type = self.dataload_config.get("media_type", "image")
        if self.media_type not in {"image", "video"}:
            raise ValueError(
                f"Unsupported media_type={self.media_type!r}; "
                "expected 'image' or 'video'."
            )
        # V1 video path: keep prompt placeholders and media tensors in the same camera order.
        self._camera_keys = [
            key
            for key in self.hf_dataset.meta.camera_keys
            if key in self._cam_key_mapping
        ]
        if not self._camera_keys:
            raise ValueError(
                f"No mapped camera keys found for repo {self.hf_dataset.meta.repo_id!r}."
            )

    def _resize_pil_image(self, img_pil, cam_key):
        orig_width, orig_height = img_pil.size
        target_size = self.data_config.resolution.get(
            self._cam_key_mapping[cam_key], -1
        )
        if target_size != -1:
            if orig_width > orig_height:
                new_width = target_size
                new_height = int(target_size * orig_height / orig_width)
            else:
                new_height = target_size
                new_width = int(target_size * orig_width / orig_height)
            img_pil = img_pil.resize((new_width, new_height))

        current_width, current_height = img_pil.size
        resized_height, resized_width = smart_resize(
            current_height,
            current_width,
            factor=self.data_config.image_factor,
            min_pixels=self.data_config.min_pixels,
            max_pixels=self.data_config.max_pixels,
        )
        resized_img = img_pil.resize((resized_width, resized_height))
        return resized_img, orig_height, orig_width, resized_height, resized_width

    @staticmethod
    def _tensor_frame_to_pil(frame):
        from PIL import Image

        frame = frame.detach().cpu()
        if frame.ndim != 3:
            raise ValueError(f"Expected a 3D frame tensor, got shape {tuple(frame.shape)}")
        if frame.shape[0] in {1, 3, 4}:
            frame = frame.permute(1, 2, 0)
        if frame.dtype != torch.uint8:
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
        return Image.fromarray(frame.numpy()).convert("RGB")

    def _vision_preprocess(self, frames):
        processed_frames = []
        for key in self._camera_keys:
            img_pil = self._tensor_frame_to_pil(frames[key].clone())
            resized_img, orig_height, orig_width, resized_height, resized_width = (
                self._resize_pil_image(img_pil, key)
            )
            processed_frames.append(resized_img)

        return processed_frames, orig_height, orig_width, resized_height, resized_width

    def _video_preprocess(self, frames):
        """V1 video path: convert each camera window to one ordered video clip."""
        processed_videos = []
        first_shape = None
        last_resize = None

        for key in self._camera_keys:
            video_tensor = frames[key]
            if video_tensor.ndim == 3:
                # Single-frame fallback keeps video mode robust for datasets without camera deltas.
                video_tensor = video_tensor.unsqueeze(0)
            if video_tensor.ndim != 4:
                raise ValueError(
                    f"Expected camera {key!r} to be [T,C,H,W] or [C,H,W], "
                    f"got shape {tuple(video_tensor.shape)}"
                )

            clip = []
            for frame in video_tensor:
                img_pil = self._tensor_frame_to_pil(frame)
                resized_img, orig_h, orig_w, resize_h, resize_w = self._resize_pil_image(
                    img_pil, key
                )
                if first_shape is None:
                    first_shape = (orig_h, orig_w)
                last_resize = (resize_h, resize_w)
                clip.append(np.array(resized_img))
            processed_videos.append(clip)

        if first_shape is None or last_resize is None:
            raise ValueError("No camera frames were available for video preprocessing.")

        orig_h, orig_w = first_shape
        resize_h, resize_w = last_resize
        return processed_videos, orig_h, orig_w, resize_h, resize_w

    def __getitem__(self, index):
        data = self._dataset[index]
        if self.media_type == "video":
            video_inputs, h, w, resize_h, resize_w = self._video_preprocess(data)
        else:
            image_inputs, h, w, resize_h, resize_w = self._vision_preprocess(data)
        agent_pos = data[self._state_key_mapping["state"]]
        action = data[self._action_key_mapping["action"]]
        frame_index = data["frame_index"]
        instruction_info = {"instruction": data["task"]}
        generate_subtask_ratio = self.data_config.generate_subtask_ratio

        complete_text, generate_subtask = get_wallx_normal_text(
            instruction_info,
            self.dataload_config.get("action_horizon", 33) - 1,
            frame_index,
            self.data_config.priority_order,
            {key: self._cam_key_mapping[key] for key in self._camera_keys},
            generate_subtask_ratio=generate_subtask_ratio,
            media_type=self.media_type,
        )
        text = process_grounding_points(
            complete_text, h, w, resize_h, resize_w, self.data_config.model_type
        )
        result = {
            "text": text,
            "action": action,
            "agent_pos": agent_pos,
            "frame_index": frame_index,
        }
        if self.media_type == "video":
            result["video_inputs"] = video_inputs
        else:
            result["image_inputs"] = image_inputs

        return result

    def __len__(self) -> int:
        return len(self._dataset)

    def _eval(self):
        self._dataset = self.val_dataset

    def _train(self):
        self._dataset = self.train_dataset

    def get_train_dataloader(self):
        """
        Get distributed training dataloader

        Args:
            rank: Current process rank
            world_size: Total number of processes
            seed: Random seed for reproducibility
        """
        self._train()

        batch_size = self.config.get("batch_size_per_gpu", 8)
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            drop_last=True,  # Ensure all processes have same number of batches
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,  # Use distributed sampler instead of shuffle=True
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config,
                self.dataload_config,
                self.normalizer_action,
                self.normalizer_propri,
                self.lerobot_config,
            ),
            pin_memory=True,  # Enable for GPU training
            persistent_workers=num_workers > 0,  # Only if num_workers > 0
            prefetch_factor=2,  # Reduce memory usage
            drop_last=True,  # Avoid incomplete batches
        )

        return dataloader, sampler

    def get_val_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """
        self._eval()

        batch_size = self.config.get(
            "eval_batch_size_per_gpu", self.config.get("batch_size_per_gpu", 8)
        )
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler for evaluation (no shuffle)
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,  # No shuffling for evaluation
            drop_last=False,  # Keep all samples for evaluation
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
            drop_last=False,
        )

        return dataloader, sampler


class DataCollator:
    # Class-level cache for processors to avoid reloading
    _processor_cache = {}
    _action_tokenizer_cache = {}

    def __init__(
        self,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
    ):
        self.config = config
        self.dataload_config = dataload_config

        self.normalizer_action = normalizer_action[0]
        self.normalizer_propri = normalizer_propri
        self.lerobot_config = lerobot_config

        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.dataset_name = self.config["data"]["lerobot_config"].get("repo_id", "")
        self.dataset_name = [self.dataset_name] * self.config["batch_size_per_gpu"]
        self.load_processor()

    def load_processor(self):
        processor_path = self.config["pretrained_wallx_path"]
        action_tokenizer_path = self.config.get("action_tokenizer_path", None)

        if (
            self.use_fast_tokenizer
            and action_tokenizer_path not in self._action_tokenizer_cache
        ):
            self._action_tokenizer_cache[action_tokenizer_path] = (
                AutoProcessor.from_pretrained(
                    action_tokenizer_path, trust_remote_code=True
                )
            )

        # Use cached processors if available
        if processor_path not in self._processor_cache:
            processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
            if self.config.get("padding_side", "left") == "left":
                processor.tokenizer.padding_side = "left"

            new_tokens = ["<|propri|>", "<|action|>"]
            processor.tokenizer.add_tokens(new_tokens)
            if self.use_fast_tokenizer and self.config.get("model_type") == "qwen2_5":
                action_tokenizer = self._action_tokenizer_cache[action_tokenizer_path]
                new_tokens = [
                    f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)
                ]
                processor.tokenizer.add_tokens(new_tokens)
                begin_idx_token = "<|action_token_0|>"
                token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
                processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
                processor.tokenizer.init_kwargs["action_token_vocab_size"] = (
                    action_tokenizer.vocab_size
                )

            self._processor_cache[processor_path] = processor

        self.processor = self._processor_cache[processor_path]

        if not self.use_fast_tokenizer:
            self.train_action_tokenizer = None
        else:
            self.train_action_tokenizer = self._action_tokenizer_cache[
                action_tokenizer_path
            ]

    @classmethod
    def _normalize(cls, action, min_stat, delta):
        """
        Normalize action data using min-max normalization.
        """
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (action - min_stat) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        return x

    def __call__(self, batch):
        additional_inputs = {}

        for key in batch[0].keys():
            if key == "agent_pos":
                agent_pos = torch.stack([item["agent_pos"] for item in batch])
                if agent_pos.dim() == 2:
                    agent_pos = agent_pos.unsqueeze(1)
                agent_pos_mask = (~torch.isnan(agent_pos)).float()
                # print("agent_pos_mask",agent_pos_mask.shape)
                agent_pos.nan_to_num_(nan=0.0)

                # if agent_pos.shape[-1] != 20:
                #     agent_pos = torch.cat(
                #         [
                #             agent_pos,
                #             torch.zeros(
                #                 agent_pos.shape[0],
                #                 agent_pos.shape[1],
                #                 20 - agent_pos.shape[-1],
                #             ),
                #         ],
                #         dim=-1,
                #     )
                #     agent_pos_mask = torch.cat(
                #         [
                #             agent_pos_mask,
                #             torch.zeros(
                #                 agent_pos_mask.shape[0],
                #                 agent_pos_mask.shape[1],
                #                 20 - agent_pos_mask.shape[-1],
                #             ),
                #         ],
                #         dim=-1,
                #     )
                agent_pos = self.normalizer_propri.normalize_data(
                    agent_pos, self.dataset_name
                )
                additional_inputs["proprioception"] = agent_pos
                additional_inputs["agent_pos_mask"] = agent_pos_mask
            elif key == "action":
                action = torch.stack([item["action"] for item in batch])
                if action.dim() == 2:
                    action = action.unsqueeze(1)
                dof_mask = (~torch.isnan(action)).float()
                action.nan_to_num_(nan=0.0)

                # if action.shape[-1] != 20:
                #     action = torch.cat(
                #         [
                #             action,
                #             torch.zeros(
                #                 action.shape[0], action.shape[1], 20 - action.shape[-1]
                #             ),
                #         ],
                #         dim=-1,
                #     )
                #     dof_mask = torch.cat(
                #         [
                #             dof_mask,
                #             torch.zeros(
                #                 dof_mask.shape[0],
                #                 dof_mask.shape[1],
                #                 20 - dof_mask.shape[-1],
                #             ),
                #         ],
                #         dim=-1,
                #     )
                action = self.normalizer_action.normalize_data(
                    action, self.dataset_name
                )
                additional_inputs["action_chunk"] = action
                additional_inputs["dof_mask"] = dof_mask
            elif key == "image_inputs":
                additional_inputs["image_inputs"] = [
                    item["image_inputs"] for item in batch
                ]
            elif key == "video_inputs":
                # V1 video path: flatten clips in batch/camera order so each
                # <|video_pad|> placeholder consumes exactly one video clip.
                additional_inputs["video_inputs"] = [
                    clip for item in batch for clip in item["video_inputs"]
                ]
            elif key == "text":
                additional_inputs["text"] = [item["text"] for item in batch]
            elif key == "frame_index":
                additional_inputs["frame_index"] = torch.stack(
                    [item["frame_index"] for item in batch]
                )
            else:
                raise NotImplementedError(
                    f"{key} input not implemented in preprocesser"
                )

        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            self.train_action_tokenizer if self.use_fast_tokenizer else None,
            [self.lerobot_config["repo_id"]] * additional_inputs["text"].__len__(),
            additional_inputs["dof_mask"],
        )

        inputs = preprocesser_call(
            processor=self.processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs", None),
            videos=additional_inputs.pop("video_inputs", None),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.dataload_config.get("max_length", 768),
        )

        # V1 video path: make temporal RoPE explicit when video_stride is configured.
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            seconds_per_grid = self.dataload_config.get("video_seconds_per_grid", None)
            if seconds_per_grid is not None and "second_per_grid_ts" not in inputs:
                inputs["second_per_grid_ts"] = torch.full(
                    (inputs["video_grid_thw"].shape[0],),
                    float(seconds_per_grid),
                    dtype=torch.float32,
                )

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Gating token types
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id

        inputs.update(additional_inputs)

        inputs["dataset_names"] = [self.lerobot_config["repo_id"]] * inputs[
            "action_chunk"
        ].shape[0]

        return inputs


def _build_lerobot_delta_timestamps(repo_id, dataset_fps, dataload_config):
    """Build action and optional V1 video camera windows for LeRobotDataset."""
    delta_timestamps = {
        KEY_MAPPINGS[repo_id]["action"]: [
            t / dataset_fps
            for t in range(dataload_config.get("action_horizon", 33) - 1)
        ],
    }

    if dataload_config.get("media_type", "image") == "video":
        num_frames = int(dataload_config.get("video_num_frames", 8))
        stride = int(dataload_config.get("video_stride", 1))
        if num_frames <= 0:
            raise ValueError(
                "data.video_num_frames must be positive for video media_type."
            )
        if stride <= 0:
            raise ValueError(
                "data.video_stride must be positive for video media_type."
            )

        # V1 video path: use a history window ending at the current dataset index.
        camera_offsets = [
            -(num_frames - 1 - i) * stride / dataset_fps for i in range(num_frames)
        ]
        for camera_key in KEY_MAPPINGS[repo_id]["camera"]:
            delta_timestamps[camera_key] = camera_offsets
        dataload_config["video_seconds_per_grid"] = stride / dataset_fps

    return delta_timestamps


def load_lerobot_data(
    config,
    lerobot_config,
    normalizer_action,
    normalizer_propri,
    rank=0,
    world_size=1,
    seed=42,
):
    """
    Load LeRobot dataset with distributed support

    Args:
        config: Model configuration
        rank: Current process rank (default: 0)
        world_size: Total number of processes (default: 1)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Training dataset
        train_num: Number of training samples per process
        sampler: Distributed sampler (None if world_size=1)
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    dataload_config = get_data_configs(config["data"])

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    episodes_num = meta_info.total_episodes

    # norm_stats_path = config.get("norm_stats_path", None)
    # assert (
    #     norm_stats_path is not None
    # ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    # norm_stats = load_norm_stats(norm_stats_path, repo_id)

    delta_timestamps = _build_lerobot_delta_timestamps(
        repo_id, dataset_fps, dataload_config
    )
    batch_size = config.get("batch_size_per_gpu", 8)
    episodes = np.arange(episodes_num).tolist()

    train_test_split = dataload_config.get("train_test_split", 0.95)
    train_episodes = episodes[: int(episodes_num * train_test_split)]
    test_episodes = episodes[int(episodes_num * train_test_split) :]

    train_dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    if rank == 0:
        print(f"Selected train episodes: {train_dataset.episodes}")
        print(f"Number of train episodes selected: {train_dataset.num_episodes}")
        print(f"Number of train frames selected: {train_dataset.num_frames}")
        print(f"Selected test episodes: {test_episodes}")

    dataset = PreprocessedDataset(
        train_dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Calculate samples per process
    if world_size > 1:
        # With DistributedSampler, each process gets approximately len(dataset) // world_size samples
        samples_per_process = len(dataset) // world_size
        train_num = samples_per_process // batch_size
    else:
        train_num = len(dataset) // batch_size

    if rank == 0:
        print("\n" + "=" * 50)
        print("LeRobot Data Loading Configuration:")
        print(f"✦ RANK: {rank}")
        print(f"✦ WORLD SIZE: {world_size}")
        print(f"✦ BATCH SIZE PER GPU: {batch_size}")
        print(f"✦ REPO ID: {repo_id}")
        print(f"✦ TOTAL DATASET SIZE: {len(dataset)}")
        if world_size > 1:
            print(f"✦ SAMPLES PER PROCESS: {samples_per_process}")
            print(f"✦ BATCHES PER PROCESS: {train_num}")
            print(f"✦ TOTAL BATCHES (ALL PROCESSES): {train_num * world_size}")
        else:
            print(f"✦ TOTAL BATCHES: {train_num}")
        print(f"✦ SEED: {seed}")
        print("=" * 50 + "\n")

    return dataset, train_num


def get_distributed_dataloader(
    dataset, config, rank=0, world_size=1, seed=42, is_train=True
):
    """
    Helper function to get distributed dataloader

    Args:
        dataset: PreprocessedDataset instance
        config: Configuration dict
        rank: Current process rank
        world_size: Total number of processes
        seed: Random seed
        is_train: Whether this is for training (affects shuffling)

    Returns:
        dataloader: Distributed DataLoader
        sampler: DistributedSampler
    """
    if is_train:
        return dataset.get_train_dataloader(rank=rank, world_size=world_size, seed=seed)
    else:
        return dataset.get_val_dataloader(rank=rank, world_size=world_size)


def get_data_configs(config):
    default_data_config = {
        "train_test_split": 0.95,
        "split_seed": 42,
        "media_type": "image",
        "video_num_frames": 8,
        "video_stride": 1,
        "batch_size": 8,
        "action_horizon": 21,
        "action_history_length": 0,
        "image_horizon": 1,
        "image_history_length": 0,
        "left_padding": False,
        "right_padding": False,
        "return_first_obs": False,
        "return_last_obs": False,
        "randomize_obs_after": None,
        "datasets": [],
        "labeled_pathes": [],
    }
    data_config = default_data_config | config
    data_config["action_horizon"] += 1

    return data_config


class TestDataset(PreprocessedDataset):
    def __init__(
        self,
        dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=42,
    ):
        super().__init__(
            dataset,
            config,
            dataload_config,
            normalizer_action,
            normalizer_propri,
            lerobot_config,
            seed=seed,
            rank=0,
            world_size=1,
            test_only=True,
        )

    def get_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=1,
            collate_fn=DataCollator(
                self.config,
                self.dataload_config,
                self.normalizer_action,
                self.normalizer_propri,
                self.lerobot_config,
            ),
        )

        return dataloader


def load_test_dataset(
    config,
    lerobot_config,
    normalizer_action,
    normalizer_propri,
    seed=42,
    episode=0,
):
    """
    Load test dataset

    Args:
        config: Model configuration
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Test dataset
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    dataload_config = get_data_configs(config["data"])

    norm_stats_path = config.get("norm_stats_path", None)
    assert (
        norm_stats_path is not None
    ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    # norm_stats = load_norm_stats(norm_stats_path, repo_id)

    delta_timestamps = _build_lerobot_delta_timestamps(
        repo_id, dataset_fps, dataload_config
    )

    dataset = LeRobotDataset(
        repo_id,
        episodes=[episode],
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        root=root,
    )

    print(f"Selected episodes: {dataset.episodes}")
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    dataset = TestDataset(
        dataset,
        config,
        dataload_config,
        normalizer_action,
        normalizer_propri,
        lerobot_config,
        seed=seed,
    )

    return dataset
