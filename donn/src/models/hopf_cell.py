import torch
import torch.nn as nn
import numpy as np


class HopfCell(nn.Module):

    def __init__(
        self,
        units=50,
        min_freq=8.0,
        max_freq=13.0,
        dt=0.01,
        beta=10.0,
        input_scaler=5.0
    ):
        super().__init__()

        self.units = units
        self.dt = dt
        self.beta = beta
        self.input_scaler = input_scaler

        # -----------------------------------------
        # Initialize oscillator frequencies
        # Mu rhythm range: 8–13 Hz
        # -----------------------------------------

        freqs = torch.linspace(
            min_freq,
            max_freq,
            units
        )

        omegas = 2 * np.pi * freqs

        self.omega = nn.Parameter(
            omegas
        )

    # --------------------------------------------------
    # initialize oscillator state
    # --------------------------------------------------

    def init_state(
        self,
        batch_size,
        device
    ):

        r = torch.ones(
            batch_size,
            self.units,
            device=device
        )

        phi = torch.zeros(
            batch_size,
            self.units,
            device=device
        )

        return r, phi

    # --------------------------------------------------
    # one oscillator update
    # --------------------------------------------------

    def forward(
        self,
        input_r,
        input_i,
        r,
        phi
    ):

        # input_r : (B,U)
        # input_i : (B,U)

        drive_r = (
            self.input_scaler
            * input_r
            * torch.cos(phi)
        )

        drive_phi = (
            self.input_scaler
            * input_i
            * torch.sin(phi)
        )

        # -----------------------------------------
        # amplitude update
        # -----------------------------------------

        r_next = r + (

            (
                (1 - self.beta * r**2)
                * r
            )
            + drive_r

        ) * self.dt

        # -----------------------------------------
        # phase update
        # -----------------------------------------

        phi_next = phi + (

            self.omega.unsqueeze(0)
            - drive_phi

        ) * self.dt

        # -----------------------------------------
        # oscillator outputs
        # -----------------------------------------

        z_real = r_next * torch.cos(
            phi_next
        )

        z_imag = r_next * torch.sin(
            phi_next
        )

        return (
            z_real,
            z_imag,
            r_next,
            phi_next
        )