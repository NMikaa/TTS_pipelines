"""
LSD (Lagrangian Self-Distillation) Loss for Pocket TTS training.

Based on the CALM paper (https://arxiv.org/abs/2505.18825).

Uses linear interpolation flow matching (consistent with the Euler
integration in the model's lsd_decode inference):
    x_t = (1-t)*noise + t*data,   t in [0, 1]
    velocity = data - noise        (constant for linear flow)

The flow_net F(c, s, t, x) predicts velocity conditioned on:
    c: transformer output conditioning
    s: start time
    t: target time
    x: current point

Loss has two components:
    1. Flow Matching (75%): F(c, t, t, x_t) should predict (data - noise)
    2. LSD (25%): F(c, s, t, x_s) should be consistent with F(c, t, t, f(x_s))
       where f(x_s) = x_s + (t-s)*F(c, s, t, x_s) is the transported point
"""

import torch
import torch.nn as nn
from typing import Optional


class LSDLoss(nn.Module):
    """Combined Flow Matching + LSD loss."""

    def __init__(self, fm_ratio: float = 0.75):
        super().__init__()
        self.fm_ratio = fm_ratio

    def _create_noised_sample(
        self,
        x_data: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1-t)*noise + t*data."""
        return (1.0 - t) * noise + t * x_data

    def flow_matching_loss(
        self,
        flow_net: nn.Module,
        x_data: torch.Tensor,
        conditioning: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Flow Matching loss.

        Args:
            flow_net: F(c, s, t, x) -> velocity prediction
            x_data: [N, D] target latents (in flow space)
            conditioning: [N, cond_dim] transformer outputs
            mask: [N] boolean mask, True = valid position

        Returns:
            loss, metrics dict
        """
        N, D = x_data.shape
        device = x_data.device

        # Sample time uniformly in (0, 1)
        t = torch.rand(N, 1, device=device).clamp(1e-4, 1.0 - 1e-4)
        noise = torch.randn_like(x_data)

        # Noised sample
        x_t = self._create_noised_sample(x_data, t, noise)

        # Target velocity (constant for linear flow)
        v_target = x_data - noise

        # Predict: F(c, t, t, x_t) for flow matching (s = t)
        v_pred = flow_net(conditioning, t, t, x_t)

        # Per-sample MSE
        mse = (v_pred - v_target).pow(2).sum(dim=-1)  # [N]

        if mask is not None:
            loss = (mse * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = mse.mean()

        return loss, {
            "fm_loss": loss.item(),
            "fm_mse": mse.mean().item(),
        }

    def lsd_loss(
        self,
        flow_net: nn.Module,
        x_data: torch.Tensor,
        conditioning: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Lagrangian Self-Distillation loss.

        Enforces consistency: F(c, s, t, x_s) ~ F(c, t, t, transported_x)

        Args:
            flow_net: F(c, s, t, x) -> velocity prediction
            x_data: [N, D] target latents
            conditioning: [N, cond_dim] transformer outputs
            mask: [N] boolean mask

        Returns:
            loss, metrics dict
        """
        N, D = x_data.shape
        device = x_data.device

        # Sample t in (0, 1) and s in (0, t)
        t = torch.rand(N, 1, device=device).clamp(1e-4, 1.0 - 1e-4)
        s = torch.rand(N, 1, device=device) * t
        s = s.clamp(min=1e-4)

        noise = torch.randn_like(x_data)

        # Create sample at time s
        x_s = self._create_noised_sample(x_data, s, noise)

        # Predict flow from s to t
        F_pred = flow_net(conditioning, s, t, x_s)

        # Transport from s to t using predicted flow
        transported = x_s + (t - s) * F_pred

        # Predict flow at transported point (with stop gradient)
        F_at_transported = flow_net(conditioning, t, t, transported.detach())

        # Consistency loss
        mse = (F_pred - F_at_transported).pow(2).sum(dim=-1)  # [N]

        if mask is not None:
            loss = (mse * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = mse.mean()

        return loss, {
            "lsd_loss": loss.item(),
            "lsd_mse": mse.mean().item(),
        }

    def forward(
        self,
        flow_net: nn.Module,
        x_data: torch.Tensor,
        conditioning: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Combined loss: fm_ratio * FM + (1 - fm_ratio) * LSD.

        Args:
            flow_net: F(c, s, t, x) -> velocity
            x_data: [N, D] target latents
            conditioning: [N, cond_dim] transformer outputs
            mask: [N] boolean mask for valid positions

        Returns:
            total_loss, metrics dict
        """
        N = x_data.shape[0]
        fm_size = max(1, int(N * self.fm_ratio))
        lsd_size = N - fm_size

        fm_loss, fm_metrics = self.flow_matching_loss(
            flow_net,
            x_data[:fm_size],
            conditioning[:fm_size],
            mask[:fm_size] if mask is not None else None,
        )

        if lsd_size > 0:
            lsd_loss_val, lsd_metrics = self.lsd_loss(
                flow_net,
                x_data[fm_size:],
                conditioning[fm_size:],
                mask[fm_size:] if mask is not None else None,
            )
        else:
            lsd_loss_val = torch.tensor(0.0, device=x_data.device)
            lsd_metrics = {}

        total_loss = self.fm_ratio * fm_loss + (1 - self.fm_ratio) * lsd_loss_val

        metrics = {
            "total_loss": total_loss.item(),
            **fm_metrics,
            **lsd_metrics,
        }

        return total_loss, metrics
