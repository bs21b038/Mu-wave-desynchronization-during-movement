from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.training.dataset import TrajectoryDataset
from src.models.hopf_model import HopfTrajectoryModel


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SEED = 42

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 1e-3

TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.1

# --------------------------------------------------
# HOPF PARAMETERS
# --------------------------------------------------

NUM_OSCILLATORS = 50

DT = 0.01

BETA = 10.0

INPUT_SCALER = 5.0

HIDDEN_DIM = 64

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


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
# DATASET
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

dataset = TrajectoryDataset(ROOT)

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = HopfTrajectoryModel(
    input_dim=6,
    num_oscillators=NUM_OSCILLATORS,
    hidden_dim=HIDDEN_DIM,
    dt=DT,
    beta=BETA,
    input_scaler=INPUT_SCALER
).to(device)

freqs = (
    model.hopf.omega.detach().cpu()
    / (2 * torch.pi)
)

print()

print(
    f"Initial frequency range: "
    f"{freqs.min():.2f} - "
    f"{freqs.max():.2f} Hz"
)

print()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_fn = nn.MSELoss()


# --------------------------------------------------
# TEACHER FORCING SCHEDULE
# --------------------------------------------------

def teacher_forcing_prob(epoch):

    frac = (
        epoch - 1
    ) / max(
        1,
        EPOCHS - 1
    )

    return (
        TEACHER_FORCING_START
        +
        (
            TEACHER_FORCING_END
            -
            TEACHER_FORCING_START
        )
        * frac
    )

mlflow.set_experiment(
    "hopf_autoregressive"
)

with mlflow.start_run():

    mlflow.log_params({

        "seed": SEED,

        "batch_size": BATCH_SIZE,

        "epochs": EPOCHS,

        "hidden_dim": HIDDEN_DIM,

        "learning_rate": LEARNING_RATE,

        "num_oscillators": NUM_OSCILLATORS,

        "dt": DT,

        "beta": BETA,

        "input_scaler": INPUT_SCALER,

        "teacher_forcing_start":
            TEACHER_FORCING_START,

        "teacher_forcing_end":
            TEACHER_FORCING_END
    })
    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        tf_prob = teacher_forcing_prob(epoch)

        # ==========================================
        # TRAIN
        # ==========================================

        model.train()

        train_loss_epoch = 0.0

        for traj_batch, target_batch, _ in train_loader:

            traj_batch = traj_batch.to(device)
            target_batch = target_batch.to(device)

            B = traj_batch.shape[0]

            optimizer.zero_grad()

            state = model.init_state(
                batch_size=B,
                device=device
            )

            prev = traj_batch[:, 0]

            loss = 0.0

            T = traj_batch.shape[1]

            for t in range(1, T):

                inp = torch.cat(
                    [
                        prev,
                        target_batch
                    ],
                    dim=1
                )

                pred, state = model.forward_step(
                    inp,
                    state
                )

                gt = traj_batch[:, t]

                loss += loss_fn(
                    pred,
                    gt
                )

                # --------------------------
                # Teacher forcing
                # --------------------------

                if random.random() < tf_prob:

                    prev = gt

                else:

                    prev = pred.detach()

            loss = loss / (T - 1)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)

        # ==========================================
        # VALIDATION
        # ==========================================

        model.eval()

        val_loss_epoch = 0.0

        with torch.no_grad():

            for traj_batch, target_batch, _ in val_loader:

                traj_batch = traj_batch.to(device)
                target_batch = target_batch.to(device)

                B = traj_batch.shape[0]

                state = model.init_state(
                    batch_size=B,
                    device=device
                )

                prev = traj_batch[:, 0]

                loss = 0.0

                T = traj_batch.shape[1]

                for t in range(1, T):

                    inp = torch.cat(
                        [
                            prev,
                            target_batch
                        ],
                        dim=1
                    )

                    pred, state = model.forward_step(
                        inp,
                        state
                    )

                    gt = traj_batch[:, t]

                    loss += loss_fn(
                        pred,
                        gt
                    )

                    # Pure autoregression
                    prev = pred

                loss = loss / (T - 1)

                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader)

        # ==========================================
        # SAVE BEST MODEL
        # ==========================================

        if val_loss_epoch < best_val_loss:

            best_val_loss = val_loss_epoch

            torch.save(
                {
                    "epoch":
                        epoch,

                    "val_loss":
                        val_loss_epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict()
                },
                ROOT / "best_hopf_model.pt"
            )

            mlflow.log_metric(
                "best_val_loss",
                best_val_loss,
                step=epoch
            )

        mlflow.log_metric(
            "train_loss",
            train_loss_epoch,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss_epoch,
            step=epoch
        )

        mlflow.log_metric(
            "teacher_forcing",
            tf_prob,
            step=epoch
        )

        # ==========================================
        # LOG
        # ==========================================

        print(
            f"Epoch {epoch:03d}"
            f" | Train {train_loss_epoch:.6f}"
            f" | Val {val_loss_epoch:.6f}"
            f" | TF {tf_prob:.3f}"
        )

    print()

    print("Training complete.")

    # -----------------------------------------
    # Save learned frequencies
    # -----------------------------------------

    freqs = (
        model.hopf.omega.detach().cpu()
        / (2 * torch.pi)
    ).numpy()

    freq_path = (
        ROOT
        / "trained_frequencies.npy"
    )

    np.save(
        freq_path,
        freqs
    )

    mlflow.log_artifact(
        str(freq_path)
    )

    # -----------------------------------------
    # Log model
    # -----------------------------------------

    mlflow.pytorch.log_model(
        model,
        artifact_path="hopf_model"
    )
        

