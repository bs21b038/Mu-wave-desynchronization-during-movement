import torch
import torch.nn as nn

from src.models.hopf_cell import HopfCell


class HopfTrajectoryModel(nn.Module):

    def __init__(
        self,
        input_dim=6,
        num_oscillators=50,
        hidden_dim=64,
        dt=0.01,
        beta=10.0,
        input_scaler=5.0
    ):
        super().__init__()

        self.num_oscillators = num_oscillators

        # -----------------------------------------
        # Input projections
        # -----------------------------------------

        self.input_r = nn.Linear(
            input_dim,
            num_oscillators
        )

        self.input_i = nn.Linear(
            input_dim,
            num_oscillators
        )

        # -----------------------------------------
        # Hopf layer
        # -----------------------------------------

        self.hopf = HopfCell(
            units=num_oscillators,
            min_freq=8.0,
            max_freq=13.0,
            dt=dt,
            beta=beta,
            input_scaler=input_scaler
        )

        # -----------------------------------------
        # Readout network
        # -----------------------------------------

        self.fc1 = nn.Linear(
            2 * num_oscillators,
            hidden_dim
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            hidden_dim,
            4
        )

        self.dropout = nn.Dropout(
            0.1
        )

    # --------------------------------------------------
    # Initialize oscillator state
    # --------------------------------------------------

    def init_state(
        self,
        batch_size,
        device
    ):

        return self.hopf.init_state(
            batch_size,
            device
        )

    # --------------------------------------------------
    # One autoregressive prediction step
    # --------------------------------------------------

    def forward_step(
        self,
        x,
        state
    ):
        """
        x:
        (B,6)

        state:
        (r, phi)

        returns:
        pred_coords : (B,4)
        new_state
        """

        r, phi = state

        # -----------------------------------------
        # Input projections
        # -----------------------------------------

        input_r = self.input_r(x)

        input_i = self.input_i(x)

        # -----------------------------------------
        # Oscillator update
        # -----------------------------------------

        (
            z_real,
            z_imag,
            r_next,
            phi_next
        ) = self.hopf(
            input_r,
            input_i,
            r,
            phi
        )

        # -----------------------------------------
        # Readout
        # -----------------------------------------

        features = torch.cat(
            [
                z_real,
                z_imag
            ],
            dim=1
        )

        hidden = self.relu(
            self.fc1(features)
        )

        hidden = self.dropout(
            hidden
        )

        pred = self.fc2(hidden)

        return (
            pred,
            (r_next, phi_next)
        )
    



'''device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

#test code
model = HopfTrajectoryModel().to(device)

state = model.init_state(
    batch_size=8,
    device=device
)

x = torch.randn(8,6).to(device)

pred, state = model.forward_step(
    x,
    state
)

print(pred.shape)
print(state[0].shape)
print(state[1].shape)

loss = pred.mean()

loss.backward()

print("Backward pass successful")

print("\nOmega shape:")
print(model.hopf.omega.shape)

print("\nFirst 5 frequencies (Hz):")

freqs = (
    model.hopf.omega.detach().cpu()
    / (2 * torch.pi)
)

print(freqs[:5])'''