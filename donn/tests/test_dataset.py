from pathlib import Path

from src.training.dataset import (
    TrajectoryDataset
)

ROOT = Path(__file__).resolve().parents[1]

dataset = TrajectoryDataset(ROOT)

traj, target, arm = dataset[0]

print()

print("Trajectory:", traj.shape)
print("Target:", target.shape)
print("Arm:", arm)

print()

print("First timestep:")
print(traj[0])

print()

print("Last timestep:")
print(traj[-1])

print()

print("Target:")
print(target)