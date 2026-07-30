"""Public structural skeleton of the lightweight depth model.

This module preserves the public architecture and method interfaces of the
V7-Lite model without disclosing the unpublished implementation. The complete
implementation will be released after the associated paper is published.
"""

from typing import Optional

import torch
import torch.nn as nn

from psd2.config.config import configurable

from .viper_model import ClipViper, META_ARCH_REGISTRY


_WITHHELD_MESSAGE = (
    "The depth-model-lite implementation is withheld until the associated "
    "paper is published."
)


class FiLMDepthAdapter(nn.Module):
    """Public interface of the lightweight depth adapter."""

    def __init__(
        self,
        channels: int,
        hidden: int,
        stats_dim: int,
        norm_type: str = "bn",
        film_init_std: float = 0.0,
    ):
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def forward(self, depth: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """Adapt depth features using image-level statistics."""
        raise NotImplementedError(_WITHHELD_MESSAGE)


class AlphaRouter(nn.Module):
    """Public interface of the lightweight fusion router."""

    def __init__(self, in_dim: int, hidden: int = 64, init_bias: float = 0.0):
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Return a sample-wise fusion weight."""
        raise NotImplementedError(_WITHHELD_MESSAGE)


class SpatialGate(nn.Module):
    """Optional spatial fusion interface retained for structural reference."""

    def __init__(self, in_channels: int, mid: int = 64):
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Return a spatial fusion mask."""
        raise NotImplementedError(_WITHHELD_MESSAGE)


@META_ARCH_REGISTRY.register()
class DepthModelLite(ClipViper):
    """Standalone structural interface of the V7-Lite model.

    The class depends directly on :class:`ClipViper`; no intermediate depth
    model is imported. The implementation will be released after publication.
    """

    @configurable
    def __init__(
        self,
        depth_fusion_norm: str = "bn",
        depth_fusion_adapter_ratio: float = 0.25,
        depth_fusion_router_hidden: int = 64,
        depth_fusion_alpha_init_bias: float = -2.0,
        depth_fusion_use_spatial_gate: bool = False,
        depth_fusion_film_init_std: float = 0.0,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(_WITHHELD_MESSAGE)

    @classmethod
    def from_config(cls, cfg):
        """Build constructor arguments from a project configuration."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def forward(self, input_list, infgt: bool = False):
        """Dispatch query, training-gallery, or inference-gallery inputs."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def _compute_degradation_stats_map(
        self, rgb_map: torch.Tensor, depth_map: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute image-level modality statistics."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def _compute_degradation_stats_patches(
        self, rgb_patches: torch.Tensor, depth_patches: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute region-level modality statistics."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def _fuse_one(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        stats: torch.Tensor,
        adapter: nn.Module,
        router: nn.Module,
        gate: Optional[nn.Module],
        tag: str,
    ) -> torch.Tensor:
        """Fuse one backbone feature stage."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def backbone(self, images: torch.Tensor, depth_list=None):
        """Extract and fuse lightweight backbone features."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def _collect_depth_patches(self, instances, positive_mask, depth_list):
        """Collect depth regions aligned with positive detections."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def _collect_rgb_patches(self, instances, positive_mask, image_list):
        """Collect RGB regions aligned with positive detections."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def _apply_post_roi_depth_fusion(
        self, box_features, depth_patches, rgb_patches
    ):
        """Apply lightweight fusion to region features."""
        raise NotImplementedError(_WITHHELD_MESSAGE)

    def forward_gallery(self, image_list, depth_list, gt_instances):
        """Run the gallery branch and return detections or training losses."""
        raise NotImplementedError(_WITHHELD_MESSAGE)
