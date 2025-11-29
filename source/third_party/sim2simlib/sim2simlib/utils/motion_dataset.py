"""Motion dataset and dataloader for loading motion data from NPZ files.

This module provides PyTorch-based Dataset and DataLoader for loading motion data
from NPZ files with support for quantity-based sampling and train/val split.
"""

import os
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class Motion_Dataset(Dataset):
    """PyTorch Dataset for loading motion data from NPZ files.
    
    This dataset loads motion data lazily (on-demand in __getitem__) to avoid
    loading all motion data into memory at once, which could cause OOM issues.
    
    Args:
        dataset_dirs: List of dataset directory paths. Each should follow the structure:
            ./datasets/npz_datasets/{dataset_name}/{robot_name}/
        robot_name: Name of the robot (e.g., "g1").
        splits: List of dataset splits corresponding to each dataset_dir. 
            Must have the same length as dataset_dirs. Each element can be:
            - A string: single split name (e.g., "train", "val", "walk_subset")
            - A list of strings: multiple splits to combine (e.g., ["train", "walk_subset"])
        
    Example:
        >>> # Single split per dataset
        >>> dataset = Motion_Dataset(
        ...     dataset_dirs=["./datasets/npz_datasets/LAFAN1_Retargeting_Dataset"],
        ...     robot_name="g1",
        ...     splits=["train"]
        ... )
        >>> 
        >>> # Multiple datasets with different splits
        >>> dataset = Motion_Dataset(
        ...     dataset_dirs=[
        ...         "./datasets/npz_datasets/LAFAN1_Retargeting_Dataset",
        ...         "./datasets/npz_datasets/LAFAN1_Retargeting_Dataset"
        ...     ],
        ...     robot_name="g1",
        ...     splits=["train", ["train", "walk_subset"]]  # Second dataset combines two splits
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Motion shape: {sample['joint_pos'].shape}")
    """
    
    def __init__(
        self,
        dataset_dirs: list[str],
        robot_name: str,
        splits: list[Union[str, list[str]]],
    ):
        """Initialize the Motion_Dataset.
        
        Args:
            dataset_dirs: List of dataset directory paths.
            robot_name: Robot name.
            splits: List of dataset splits, must be the same length as dataset_dirs.
                Each element corresponds to the dataset at the same index in dataset_dirs.
                Can be a string (single split) or list of strings (multiple splits to combine).
            
        Raises:
            ValueError: If splits and dataset_dirs have different lengths.
            FileNotFoundError: If dataset directory or info file (info.yaml/info.yml) doesn't exist.
        """
        super().__init__()
        
        # Validate that splits and dataset_dirs have the same length
        if len(splits) != len(dataset_dirs):
            raise ValueError(
                f"Length of splits ({len(splits)}) must match length of dataset_dirs ({len(dataset_dirs)})"
            )
        
        self.dataset_dirs = [Path(d).expanduser().resolve() for d in dataset_dirs]
        self.robot_name = robot_name
        self.splits = splits
        
        # Storage for NPZ file paths and metadata
        self.npz_paths: list[Path] = []
        self.quantities: list[int] = []  # Quality/difficulty of each motion clip
        self.motion_names: list[str] = []  # Base name of each motion
        self.dataset_sources: list[str] = []  # Track which dataset each motion comes from
        
        # Load dataset information and collect NPZ paths
        self._load_dataset_info()
        
        print(f"[Motion_Dataset] Loaded {len(self.npz_paths)} motion clips from {len(self.dataset_dirs)} dataset(s)")
        print(f"[Motion_Dataset] Quantity distribution: {self._get_quantity_stats()}")
    
    def _load_dataset_info(self):
        """Load dataset information from info.yaml or info.yml files and collect NPZ paths."""
        for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
            split_config = self.splits[dataset_idx]
            
            # Normalize split_config to always be a list
            if isinstance(split_config, str):
                split_names = [split_config]
            else:
                split_names = split_config
            
            # Try YAML files only (info.yaml or info.yml)
            info_path = None
            for ext in [".yaml", ".yml"]:
                candidate_path = dataset_dir / f"info{ext}"
                if candidate_path.exists():
                    info_path = candidate_path
                    break
            
            if info_path is None:
                raise FileNotFoundError(
                    f"Dataset info file not found in {dataset_dir}. "
                    f"Expected: info.yaml or info.yml"
                )
            
            # Load dataset info from YAML
            with open(info_path, "r") as f:
                info = yaml.safe_load(f)
            
            dataset_name = info["dataset"]
            
            # Process each split in the configuration
            for split in split_names:
                split_info = info.get(split, {})
                
                if not split_info:
                    raise ValueError(f"[Motion_Dataset] No '{split}' data in {dataset_name}")
                
                # Construct path to robot-specific NPZ files
                robot_dir = dataset_dir / self.robot_name
                
                if not robot_dir.exists():
                    raise FileNotFoundError(f"Robot directory not found: {robot_dir}")
                
                # Collect NPZ paths for this split
                for motion_name, quantity in split_info.items():
                    npz_path = robot_dir / f"{motion_name}.npz"
                    
                    if npz_path.exists():
                        self.npz_paths.append(npz_path)
                        self.quantities.append(quantity)
                        self.motion_names.append(motion_name)
                        # Record source as dataset:split1+split2+... for combined splits
                        split_str = "+".join(split_names) if len(split_names) > 1 else split_names[0]
                        self.dataset_sources.append(f"{dataset_name}:{split_str}")
                    else:
                        print(f"[Motion_Dataset] Warning: NPZ file not found: {npz_path}")
    
    def _get_quantity_stats(self) -> dict[int, int]:
        """Get statistics of quantity distribution.
        
        Returns:
            Dictionary mapping quantity to count.
        """
        stats = {}
        for q in self.quantities:
            stats[q] = stats.get(q, 0) + 1
        return stats
    
    def __len__(self) -> int:
        """Return the number of motion clips in the dataset.
        
        Returns:
            Number of motion clips.
        """
        return len(self.npz_paths)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return a single motion clip.
        
        This method loads the NPZ file on-demand to avoid memory issues.
        
        Args:
            idx: Index of the motion clip to load.
            
        Returns:
            Dictionary containing:
                - motion: Dictionary of numpy arrays with motion data:
                    - joint_pos: (num_frames, num_joints) joint positions
                    - joint_vel: (num_frames, num_joints) joint velocities
                    - body_pos_w: (num_frames, num_bodies, 3) body positions
                    - body_quat_w: (num_frames, num_bodies, 4) body quaternions
                    - body_lin_vel_w: (num_frames, num_bodies, 3) body linear velocities
                    - body_ang_vel_w: (num_frames, num_bodies, 3) body angular velocities
                - fps: Frames per second of the motion data
                - length: Number of frames in the motion
                - duration: Duration in seconds
                - npz_path: Path to the NPZ file
                - motion_name: Name of the motion
                - quantity: Quality/difficulty rating (1: best, 2: medium, 3: hard)
                - dataset_source: Source dataset and split (format: "dataset_name:split")
        """
        npz_path = self.npz_paths[idx]
        
        # Load motion data
        data = np.load(npz_path)
        
        # Extract motion data
        motion = {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
        }
        
        # Add contact data if available
        # if "contact" in data:
            # motion["contact"] = data["contact"]
        
        fps = int(data["fps"][0])
        length = motion["joint_pos"].shape[0]
        duration = (length - 1) / fps
        
        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "duration": duration,
            "npz_path": str(npz_path),
            "motion_name": self.motion_names[idx],
            "quantity": self.quantities[idx],
            "dataset_source": self.dataset_sources[idx],
        }
    
    def get_motion_info(self) -> list[dict[str, Any]]:
        """Get information about all motions without loading the full data.
        
        Returns:
            List of dictionaries containing motion metadata.
        """
        info_list = []
        for i in range(len(self)):
            # Load only to get metadata (could be optimized to cache this)
            data = np.load(self.npz_paths[i])
            fps = int(data["fps"][0])
            length = data["joint_pos"].shape[0]
            
            info_list.append({
                "index": i,
                "motion_name": self.motion_names[i],
                "npz_path": str(self.npz_paths[i]),
                "quantity": self.quantities[i],
                "fps": fps,
                "length": length,
                "duration": (length - 1) / fps,
                "dataset_source": self.dataset_sources[i],
            })
        
        return info_list
    
    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics.
        """
        total_frames = 0
        total_duration = 0.0
        lengths = []
        
        for i in range(len(self)):
            data = np.load(self.npz_paths[i])
            fps = int(data["fps"][0])
            length = data["joint_pos"].shape[0]
            duration = (length - 1) / fps
            
            total_frames += length
            total_duration += duration
            lengths.append(length)
        
        return {
            "num_clips": len(self),
            "total_frames": total_frames,
            "total_duration": total_duration,
            "avg_frames_per_clip": total_frames / len(self) if len(self) > 0 else 0,
            "avg_duration_per_clip": total_duration / len(self) if len(self) > 0 else 0,
            "min_frames": min(lengths) if lengths else 0,
            "max_frames": max(lengths) if lengths else 0,
            "quantity_distribution": self._get_quantity_stats(),
        }


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Motion_Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset with single split
  python motion_dataset.py --dataset_dirs ./datasets/npz_datasets/LAFAN1_Retargeting_Dataset --splits train
  
  # Multiple datasets with single splits each
  python motion_dataset.py --dataset_dirs dataset1 dataset2 --splits train test
  
  # Multiple datasets with combined splits (use --splits_combined)
  python motion_dataset.py --dataset_dirs dataset1 dataset2 --splits train --splits_combined train+walk_subset
  
Note: Use '+' to separate multiple splits for a single dataset (e.g., train+walk_subset)
        """
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        default=["./datasets/npz_datasets/LAFAN1_Retargeting_Dataset"],
        help="Dataset directory paths",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="g1",
        help="Robot name",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits (must match length of dataset_dirs). Use '+' to combine multiple splits (e.g., train+walk_subset)",
    )
    args = parser.parse_args()
    
    # Parse splits: convert "train+walk_subset" format to ["train", "walk_subset"]
    parsed_splits = []
    for split_str in args.splits:
        if "+" in split_str:
            # Multiple splits combined
            parsed_splits.append(split_str.split("+"))
        else:
            # Single split
            parsed_splits.append(split_str)
    
    # Validate arguments
    if len(parsed_splits) != len(args.dataset_dirs):
        parser.error(f"Number of splits ({len(parsed_splits)}) must match number of dataset_dirs ({len(args.dataset_dirs)})")
    
    # Create dataset
    dataset = Motion_Dataset(
        dataset_dirs=args.dataset_dirs,
        robot_name=args.robot_name,
        splits=parsed_splits,
    )
    
    # Print dataset info
    print(f"\nDataset size: {len(dataset)}")
    print("\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load and display first sample
    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset[0]
        print(f"  Motion name: {sample['motion_name']}")
        print(f"  FPS: {sample['fps']}")
        print(f"  Length: {sample['length']} frames")
        print(f"  Duration: {sample['duration']:.2f} seconds")
        print(f"  Quantity: {sample['quantity']}")
        print(f"  NPZ path: {sample['npz_path']}")
        print(f"  Dataset source: {sample['dataset_source']}")
        print("\n  Motion data shapes:")
        for key, value in sample['motion'].items():
            print(f"    {key}: {value.shape}")
