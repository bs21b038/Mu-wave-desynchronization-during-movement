import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "raw"
    / "splined_trajectories_3.txt"
)

print("Loading:", DATA_FILE)

# --------------------------------------------------
# LOAD TRAJECTORIES
# --------------------------------------------------

with open(DATA_FILE, "rb") as f:
    trajectories = pickle.load(f)

print("Number of trajectories:", len(trajectories))


# --------------------------------------------------
# HOME POSITIONS
# --------------------------------------------------

LEFT_HOME = np.array([-0.15, 0.0])
RIGHT_HOME = np.array([0.15, 0.0])


# --------------------------------------------------
# EXTRACT TARGETS
# --------------------------------------------------

targets = []
active_arms = []

for traj in trajectories:

    traj = np.asarray(traj)

    left_final = traj[-1, :2]
    right_final = traj[-1, 2:]

    left_move = np.linalg.norm(
        left_final - LEFT_HOME
    )

    right_move = np.linalg.norm(
        right_final - RIGHT_HOME
    )

    # ----------------------------------
    # active arm
    # ----------------------------------

    if left_move > right_move:

        target = left_final
        active_arm = 0

    else:

        target = right_final
        active_arm = 1

    targets.append(target)
    active_arms.append(active_arm)

targets = np.asarray(targets)
active_arms = np.asarray(active_arms)


# --------------------------------------------------
# SAVE TARGETS
# --------------------------------------------------

processed_dir = (
    ROOT
    / "data"
    / "processed"
)

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    processed_dir / "targets.pkl",
    "wb"
) as f:

    pickle.dump(targets, f)

with open(
    processed_dir / "active_arms.pkl",
    "wb"
) as f:

    pickle.dump(active_arms, f)

print("Saved targets.pkl")
print("Saved active_arms.pkl")


# --------------------------------------------------
# STATS
# --------------------------------------------------

print()

print(
    "Left trajectories:",
    np.sum(active_arms == 0)
)

print(
    "Right trajectories:",
    np.sum(active_arms == 1)
)

print()

print(
    "Target X range:",
    targets[:, 0].min(),
    targets[:, 0].max()
)

print(
    "Target Y range:",
    targets[:, 1].min(),
    targets[:, 1].max()
)


# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 10))


# --------------------------------------------------
# PLOT ACTIVE ARM TRAJECTORIES
# --------------------------------------------------

for traj, arm in zip(
    trajectories,
    active_arms
):

    traj = np.asarray(traj)

    if arm == 0:

        x = traj[:, 0]
        y = traj[:, 1]

        plt.plot(
            x,
            y,
            color="blue",
            alpha=0.15
        )

    else:

        x = traj[:, 2]
        y = traj[:, 3]

        plt.plot(
            x,
            y,
            color="red",
            alpha=0.15
        )


# --------------------------------------------------
# TARGETS
# --------------------------------------------------

left_targets = targets[
    active_arms == 0
]

right_targets = targets[
    active_arms == 1
]

plt.scatter(
    left_targets[:, 0],
    left_targets[:, 1],
    color="blue",
    s=20,
    label="Left Targets"
)

plt.scatter(
    right_targets[:, 0],
    right_targets[:, 1],
    color="red",
    s=20,
    label="Right Targets"
)


# --------------------------------------------------
# AXES
# --------------------------------------------------

plt.xlim(-0.8, 0.8)
plt.ylim(-0.1, 0.8)

plt.xticks(
    np.arange(-0.8, 0.81, 0.1)
)

plt.yticks(
    np.arange(-0.1, 0.81, 0.1)
)

plt.grid(True)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    "Trajectory Geometry and Targets"
)

plt.legend()

plt.gca().set_aspect("equal")

plt.show()