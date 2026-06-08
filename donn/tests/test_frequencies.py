from pathlib import Path
import torch
import numpy as np

from src.models.hopf_model import HopfTrajectoryModel

ROOT = Path(__file__).resolve().parents[1]

checkpoint = torch.load(
    ROOT / "best_hopf_model.pt",
    map_location="cpu"
)

model = HopfTrajectoryModel(
    input_dim=6,
    num_oscillators=50,
    hidden_dim=64,
    dt=0.01,
    beta=10.0,
    input_scaler=5.0
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

freqs = (
    model.hopf.omega.detach()
    / (2 * torch.pi)
).numpy()

print("\nLearned frequencies (Hz):\n")
print(freqs)

print("\nMin:", freqs.min())
print("Max:", freqs.max())
print("Mean:", freqs.mean())