from __future__ import annotations

from typing import Optional

import torch


class GradientProjector:
    """Utility helpers for maintaining and applying a PEGP-style projection basis."""

    @staticmethod
    def empty_basis(feature_dim: int, device: torch.device) -> torch.Tensor:
        return torch.empty(feature_dim, 0, device=device)

    @staticmethod
    def orthonormalize(basis: Optional[torch.Tensor]) -> torch.Tensor:
        if basis is None or basis.numel() == 0:
            feature_dim = int(basis.shape[0]) if basis is not None and basis.ndim == 2 else 0
            device = basis.device if basis is not None else torch.device('cpu')
            return GradientProjector.empty_basis(feature_dim, device)

        q, _ = torch.linalg.qr(basis, mode='reduced')
        return q.contiguous()

    @staticmethod
    def estimate_basis(
        feature_sum: torch.Tensor,
        feature_outer_sum: torch.Tensor,
        sample_count: int,
        energy_threshold: float,
        max_rank: int,
        min_rank: int,
    ) -> torch.Tensor:
        feature_dim = int(feature_sum.shape[0])
        if sample_count <= 1 or max_rank <= 0:
            return GradientProjector.empty_basis(feature_dim, feature_sum.device)

        mean = feature_sum / float(sample_count)
        covariance = feature_outer_sum / float(sample_count) - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        positive = eigenvalues > 1e-8
        if not bool(torch.any(positive)):
            return GradientProjector.empty_basis(feature_dim, feature_sum.device)

        eigenvalues = eigenvalues[positive]
        eigenvectors = eigenvectors[:, positive]
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        total_energy = eigenvalues.sum().clamp_min(1e-12)
        cumulative_energy = torch.cumsum(eigenvalues, dim=0) / total_energy
        threshold = torch.tensor(float(energy_threshold), device=cumulative_energy.device)
        rank = int(torch.searchsorted(cumulative_energy, threshold, right=False).item()) + 1
        rank = max(int(min_rank), rank)
        rank = min(int(max_rank), rank, int(eigenvectors.shape[1]))

        return GradientProjector.orthonormalize(eigenvectors[:, :rank])

    @staticmethod
    def merge_bases(
        existing_basis: Optional[torch.Tensor],
        new_basis: Optional[torch.Tensor],
        max_rank: int,
    ) -> torch.Tensor:
        if new_basis is None or new_basis.numel() == 0:
            if existing_basis is None:
                return torch.empty(0, 0)
            return existing_basis.contiguous()

        if existing_basis is None or existing_basis.numel() == 0:
            merged = GradientProjector.orthonormalize(new_basis)
        else:
            residual = new_basis - existing_basis @ (existing_basis.transpose(0, 1) @ new_basis)
            merged = GradientProjector.orthonormalize(torch.cat([existing_basis, residual], dim=1))

        if merged.numel() == 0:
            return merged

        if merged.shape[1] > int(max_rank):
            merged = merged[:, : int(max_rank)]

        return merged.contiguous()

    @staticmethod
    def project_linear_weight_grad(grad: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
        if basis is None or basis.numel() == 0:
            return grad

        if grad.ndim != 2 or grad.shape[1] != basis.shape[0]:
            return grad

        return grad - (grad @ basis) @ basis.transpose(0, 1)