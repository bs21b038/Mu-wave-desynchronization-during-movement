import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, root_dir):

        root_dir = Path(root_dir)

        # -----------------------------------------
        # trajectories
        # -----------------------------------------

        traj_file = (
            root_dir
            / "data"
            / "raw"
            / "splined_trajectories_3.txt"
        )

        with open(traj_file, "rb") as f:
            self.trajectories = pickle.load(f)

        # -----------------------------------------
        # targets
        # -----------------------------------------

        target_file = (
            root_dir
            / "data"
            / "processed"
            / "targets.pkl"
        )

        with open(target_file, "rb") as f:
            self.targets = pickle.load(f)

        # -----------------------------------------
        # active arms
        # -----------------------------------------

        arm_file = (
            root_dir
            / "data"
            / "processed"
            / "active_arms.pkl"
        )

        with open(arm_file, "rb") as f:
            self.active_arms = pickle.load(f)

        # sanity check

        assert len(self.trajectories) == len(self.targets)
        assert len(self.targets) == len(self.active_arms)

        print(
            f"Loaded {len(self.trajectories)} trajectories"
        )

    def __len__(self):

        return len(self.trajectories)

    def __getitem__(self, idx):

        traj = np.asarray(
            self.trajectories[idx],
            dtype=np.float32
        )

        target = np.asarray(
            self.targets[idx],
            dtype=np.float32
        )

        active_arm = np.int64(
            self.active_arms[idx]
        )

        return (
            torch.from_numpy(traj),
            torch.from_numpy(target),
            torch.tensor(active_arm)
        )