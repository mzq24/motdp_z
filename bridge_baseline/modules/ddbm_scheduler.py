"""
DDBM Scheduler — Denoising Diffusion Bridge Model

Ported from BridgeDrive:
  BridgeDrive/BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py
  lines 44-123

Forward bridge:  x0 (GT) → xT (anchor)
Reverse bridge:  xT (anchor) → x0 (GT)

Parameterization: VP (Variance Preserving) schedule
"""

from typing import Tuple, Union
import torch


DEFAULT_BETA_D = 2.0
DEFAULT_BETA_MIN = 0.1
DEFAULT_T = 1.0
T_NORMALIZE = 1000.0


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    for _ in range(dims_to_append):
        x = x.unsqueeze(-1)
    return x


class DDBMScheduler:
    """Denoising Diffusion Bridge Model scheduler.

    Bridges between x0 (GT trajectory) and xT (anchor trajectory) via a
    Brownian Bridge process, ensuring symmetric forward/reverse processes.

    Forward:  sample(t) = a_t * xT + b_t * x0 + c_t * noise
    Reverse:  sample_step() → deterministic DDIM-style bridge step
    """

    def __init__(
        self,
        beta_d: float = DEFAULT_BETA_D,
        beta_min: float = DEFAULT_BETA_MIN,
        T: float = DEFAULT_T,
    ):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.T = T

    def vp_logs(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return -0.25 * t ** 2 * self.beta_d - 0.5 * t * self.beta_min

    def vp_logsnr(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return -torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def get_abc(
        self, t: Union[float, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bridge coefficients a_t, b_t, c_t for timestep t in [0, 1]."""
        logsnr_t = self.vp_logsnr(t)
        logsnr_T = self.vp_logsnr(self.T)
        logs_t = self.vp_logs(t)
        logs_T = self.vp_logs(self.T)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()

        return a_t, b_t, c_t

    def add_noise(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        xT: torch.Tensor,
        noise: torch.Tensor,
        t_normalize: float = T_NORMALIZE,
    ) -> torch.Tensor:
        """Forward bridge: mix GT (x0) and anchor (xT) with noise at timestep t.

        At t=0: sample ≈ x0 (GT)
        At t=T: sample ≈ xT (anchor) + noise

        Args:
            t: timestep tensor, shape (B,), values in [1, T_NORMALIZE]
            x0: clean GT trajectory, shape (B, M, T, 2) [normalized]
            xT: anchor trajectory, shape (B, M, T, 2) [normalized]
            noise: Gaussian noise, same shape as x0
            t_normalize: divisor to map t → [0, 1]
        """
        t = append_dims(t, x0.ndim) / t_normalize
        a_t, b_t, c_t = self.get_abc(t)
        return a_t * xT + b_t * x0 + c_t * noise

    def sample_step(
        self,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        xT: torch.Tensor,
        t_normalize: float = T_NORMALIZE,
    ) -> torch.Tensor:
        """Reverse bridge step: from xt at timestep t to xt_prev at timestep t_prev.

        Args:
            t: current timestep, shape (B,)
            t_prev: previous (smaller) timestep, shape (B,)
            xt: current sample [normalized], shape (B, M, T, 2)
            x0: predicted clean sample [normalized], shape (B, M, T, 2)
            xT: anchor [normalized], shape (B, M, T, 2)
        """
        is_T = ((t / t_normalize) == self.T).all()
        t = append_dims(t, x0.ndim) / t_normalize
        t_prev = append_dims(t_prev, x0.ndim) / t_normalize

        a_t, b_t, c_t = self.get_abc(t)
        a_t_prev, b_t_prev, c_t_prev = self.get_abc(t_prev)

        xt_prev = a_t_prev * xT + b_t_prev * x0
        if is_T:
            # First step: inject fresh noise
            xt_prev = xt_prev + c_t_prev * torch.randn_like(xt_prev)
        else:
            # Subsequent steps: propagate residual noise
            xt_prev = xt_prev + (c_t_prev / c_t) * (xt - a_t * xT - b_t * x0)

        return xt_prev
