from pathlib import Path
import random

import torch
import matplotlib.pyplot as plt

from src.models.hopf_model import HopfTrajectoryModel
from src.training.dataset import TrajectoryDataset


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# --------------------------------------------------
# ROOT
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# DATASET
# --------------------------------------------------

dataset = TrajectoryDataset(ROOT)

print(
    f"Loaded {len(dataset)} trajectories"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

checkpoint = torch.load(
    ROOT / "best_hopf_model.pt",
    map_location=device
)

model = HopfTrajectoryModel(
    input_dim=6,
    num_oscillators=50,
    hidden_dim=64,
    dt=0.01,
    beta=10.0,
    input_scaler=5.0
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded best model")

# --------------------------------------------------
# RANDOM SAMPLES
# --------------------------------------------------

random.seed(42)

sample_indices = random.sample(
    range(len(dataset)),
    8
)

print()

print("Sample indices:")
print(sample_indices)

# --------------------------------------------------
# FIGURE
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    4,
    figsize=(18, 8)
)

axes = axes.flatten()

# --------------------------------------------------
# LOOP OVER SAMPLES
# --------------------------------------------------

for ax, sample_idx in zip(
    axes,
    sample_indices
):

    traj, target, active_arm = dataset[sample_idx]

    traj = traj.to(device)
    target = target.to(device)

    T = traj.shape[0]

    # ------------------------------------------
    # Initialize oscillator state
    # ------------------------------------------

    state = model.init_state(
        batch_size=1,
        device=device
    )

    # ------------------------------------------
    # First coordinate
    # ------------------------------------------

    prev = traj[0].unsqueeze(0)

    predictions = []

    # ------------------------------------------
    # Pure autoregressive rollout
    # ------------------------------------------

    with torch.no_grad():

        for t in range(1, T):

            inp = torch.cat(
                [
                    prev,
                    target.unsqueeze(0)
                ],
                dim=1
            )

            pred, state = model.forward_step(
                inp,
                state
            )

            predictions.append(
                pred.squeeze(0).cpu()
            )

            prev = pred

    predictions = torch.stack(
        predictions
    )

    gt = traj[1:].cpu()

    target_cpu = target.cpu()

    # ------------------------------------------
    # Plot left arm
    # ------------------------------------------

    ax.plot(
        gt[:, 0],
        gt[:, 1],
        linewidth=2,
        label="GT Left"
    )

    ax.plot(
        predictions[:, 0],
        predictions[:, 1],
        "--",
        linewidth=2,
        label="Pred Left"
    )

    # ------------------------------------------
    # Plot right arm
    # ------------------------------------------

    ax.plot(
        gt[:, 2],
        gt[:, 3],
        linewidth=2,
        label="GT Right"
    )

    ax.plot(
        predictions[:, 2],
        predictions[:, 3],
        "--",
        linewidth=2,
        label="Pred Right"
    )

    # ------------------------------------------
    # Target
    # ------------------------------------------

    ax.scatter(
        target_cpu[0],
        target_cpu[1],
        marker="^",
        s=120,
        label="Target"
    )

    # ------------------------------------------
    # Home positions
    # ------------------------------------------

    ax.scatter(
        traj[0, 0].cpu(),
        traj[0, 1].cpu(),
        marker="o",
        s=40
    )

    ax.scatter(
        traj[0, 2].cpu(),
        traj[0, 3].cpu(),
        marker="o",
        s=40
    )

    # ------------------------------------------
    # Labels
    # ------------------------------------------

    arm_name = (
        "LEFT"
        if active_arm.item() == 0
        else "RIGHT"
    )

    ax.set_title(
        f"Sample {sample_idx}\nActive Arm: {arm_name}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid(True)

    ax.axis("equal")

# --------------------------------------------------
# LEGEND
# --------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper right"
)

plt.suptitle(
    "Hopf Pure Autoregressive Rollouts",
    fontsize=16
)

plt.tight_layout()

plt.show()