from pathlib import Path
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.hopf_model import HopfTrajectoryModel
from src.training.dataset import TrajectoryDataset


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ROOT = Path(__file__).resolve().parents[1]

dataset = TrajectoryDataset(ROOT)

# -----------------------------------------
# Load model
# -----------------------------------------

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

# -----------------------------------------
# Choose samples
# -----------------------------------------

random.seed(42)

sample_indices = random.sample(
    range(len(dataset)),
    8
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16,8)
)

axes = axes.flatten()

# -----------------------------------------
# Loop
# -----------------------------------------

for ax, idx in zip(
    axes,
    sample_indices
):

    traj, target, active_arm = dataset[idx]

    traj = traj.to(device)
    target = target.to(device)

    T = traj.shape[0]

    state = model.init_state(
        batch_size=1,
        device=device
    )

    prev = traj[0].unsqueeze(0)

    predictions = []

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

    target_np = target.cpu().numpy()

    # -------------------------------------
    # Select moving arm
    # -------------------------------------

    if active_arm.item() == 0:

        gt_arm = gt[:, :2].numpy()

        pred_arm = predictions[:, :2].numpy()

        arm_name = "LEFT"

    else:

        gt_arm = gt[:, 2:].numpy()

        pred_arm = predictions[:, 2:].numpy()

        arm_name = "RIGHT"

    gt_dist = np.linalg.norm(
        gt_arm - target_np,
        axis=1
    )

    pred_dist = np.linalg.norm(
        pred_arm - target_np,
        axis=1
    )

    ax.plot(
        gt_dist,
        label="GT"
    )

    ax.plot(
        pred_dist,
        "--",
        label="Prediction"
    )

    ax.set_title(
        f"Sample {idx}\n{arm_name}"
    )

    ax.set_xlabel("Timestep")

    ax.set_ylabel("Distance")

    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper right"
)

plt.suptitle(
    "Distance To Target During Reach"
)

plt.tight_layout()

plt.show()