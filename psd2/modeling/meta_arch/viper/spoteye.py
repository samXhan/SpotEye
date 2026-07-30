"""Public structural skeleton of the SpotEye model.

The core fusion implementation is intentionally withheld during review to
protect unpublished research. The complete implementation will be released
after the associated paper is published.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

from psd2.config.config import configurable
from psd2.modeling.losses.hard_negative_queue import (
    add_hard_negative_queue_loss,
    build_roi_queue_metadata,
    expand_queue_metadata,
)
from psd2.modeling.poolers import ROIPooler
from psd2.structures import ImageList
from psd2.utils.events import get_event_storage

from .viper_model import ClipViper, META_ARCH_REGISTRY


def _make_norm2d(channels: int, norm_type: str) -> nn.Module:
    return nn.GroupNorm(1, channels) if str(norm_type).lower() == "ln" else nn.BatchNorm2d(channels)


def _sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kx, ky


def _laplacian_kernel(device, dtype):
    return torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x
    if x.shape[1] >= 3:
        return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def _depth_stats(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)
    valid = torch.isfinite(x) & (x > 0)
    clean = torch.where(valid, x, torch.zeros_like(x))
    denom = valid.flatten(1).float().sum(dim=1).clamp_min(1.0)
    mean = clean.flatten(1).sum(dim=1) / denom
    var = ((clean - mean[:, None, None, None]) ** 2).flatten(1).sum(dim=1) / denom
    kx, ky = _sobel_kernels(x.device, x.dtype)
    gx, gy = F.conv2d(clean, kx, padding=1), F.conv2d(clean, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + eps).flatten(1).sum(dim=1) / denom
    lap = F.conv2d(clean, _laplacian_kernel(x.device, x.dtype), padding=1)
    uncertainty = 1.0 - valid.flatten(1).float().mean(dim=1) + lap.abs().flatten(1).sum(dim=1) / denom
    return torch.stack([mean, var, edge, uncertainty], dim=1)


def _rgb_stats(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gray = _to_gray(x)
    mean = gray.flatten(1).mean(dim=1)
    var = gray.flatten(1).var(dim=1, unbiased=False)
    kx, ky = _sobel_kernels(gray.device, gray.dtype)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + eps).flatten(1).mean(dim=1)
    lap = F.conv2d(gray, _laplacian_kernel(gray.device, gray.dtype), padding=1)
    lap = lap.abs().flatten(1).mean(dim=1)
    high_freq = (gray - F.avg_pool2d(gray, 3, stride=1, padding=1)).abs().flatten(1).mean(dim=1)
    return torch.stack([mean, var, grad, lap, high_freq], dim=1)


def _safe_nan_quantile(x: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    if hasattr(torch, "nanquantile"):
        return torch.nanquantile(x, q, dim=dim)
    mask = torch.isfinite(x)
    clean = torch.where(mask, x, torch.zeros_like(x))
    return clean.sum(dim=dim) / mask.float().sum(dim=dim).clamp_min(1.0)


def _depth_signature(depth_patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = torch.isfinite(depth_patches[:, 0])
    pixels = depth_patches[:, 0].clone()
    pixels[~valid] = float("nan")
    flat = pixels.flatten(1)
    median = _safe_nan_quantile(flat, 0.5, dim=1).clamp_min(eps)
    normalized = depth_patches[:, 0] / (median[:, None, None] + eps)
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    quantile_input = normalized.masked_fill(~valid, float("nan")).flatten(1)
    iqr = (
        _safe_nan_quantile(quantile_input, 0.75, 1)
        - _safe_nan_quantile(quantile_input, 0.25, 1)
    ).clamp_min(0.0)
    image = normalized.unsqueeze(1)
    kx, ky = _sobel_kernels(image.device, image.dtype)
    gx, gy = F.conv2d(image, kx, padding=1), F.conv2d(image, ky, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + eps)
    lap = F.conv2d(image, _laplacian_kernel(image.device, image.dtype), padding=1).abs()
    mask = valid.unsqueeze(1).float()
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return torch.stack(
        [
            torch.log(median),
            valid.flatten(1).float().mean(dim=1),
            torch.log1p(iqr),
            (grad * mask).flatten(1).sum(dim=1) / denom,
            (lap * mask).flatten(1).sum(dim=1) / denom,
        ],
        dim=1,
    )


def _depth_aware_supcon(emb, pid, depth_sig, alpha=2.0, tau=0.07, eps=1e-6):
    keep = pid >= 0
    emb, pid, depth_sig = emb[keep], pid[keep], depth_sig[keep]
    if emb.shape[0] <= 1:
        return emb.sum() * 0.0
    emb = F.normalize(emb, dim=1)
    z = (depth_sig - depth_sig.mean(0, keepdim=True)) / depth_sig.std(0, keepdim=True).clamp_min(eps)
    z = z * z.new_tensor([0.5, 0.25, 1.0, 2.0, 2.0])[None]
    weight = torch.exp(-alpha * torch.cdist(z, z)).clamp_min(0.05)
    logits = emb @ emb.t() / tau
    eye = torch.eye(len(emb), device=emb.device, dtype=torch.bool)
    positive = (pid[:, None] == pid[None]) & ~eye
    if not positive.any():
        return emb.sum() * 0.0
    negative = ~positive & ~eye
    logits = logits.masked_fill(eye, -1e9)
    logits = torch.where(negative, logits * weight, logits)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return (-(log_prob * positive).sum(1) / positive.sum(1).clamp_min(1)).mean()


class FiLMDepthAdapter(nn.Module):
    def __init__(self, channels, hidden, stats_dim, norm_type="bn", film_init_std=0.0):
        super().__init__()
        self.channels = int(channels)
        self.norm = _make_norm2d(channels, norm_type)
        self.proj1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.proj2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.film = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, max(16, hidden)),
            nn.GELU(),
            nn.Linear(max(16, hidden), 2 * channels),
        )
        if film_init_std > 0:
            nn.init.normal_(self.film[-1].weight, std=float(film_init_std))
        else:
            nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x, stats):
        hidden = self.proj2(self.act(self.proj1(self.norm(x))))
        gamma, beta = self.film(stats).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).view(-1, self.channels, 1, 1).to(hidden.dtype)
        beta = beta.view(-1, self.channels, 1, 1).to(hidden.dtype)
        return gamma * hidden + beta


class AlphaRouter(nn.Module):
    def __init__(self, stats_dim, hidden=64, init_bias=0.0):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(stats_dim), nn.Linear(stats_dim, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, float(init_bias))

    def forward(self, stats):
        return torch.sigmoid(self.head(self.encoder(stats)))


class DeterministicUFusion(nn.Module):
    """Interface of the unpublished SpotEye fusion module.

    The complete implementation will be released after paper publication.
    """

    def __init__(
        self,
        channels,
        stats_dim,
        norm_type="bn",
        base_ratio=0.25,
        aug_ratio=0.5,
        gate_mid=64,
        mixer_hidden=32,
        mixer_init_bias=-1.0,
        base_film_init_std=0.01,
        aug_film_init_std=0.005,
        window_size=5,
        u_alpha_init=4.0,
        u_beta_init=0.5,
    ):
        raise NotImplementedError(
            "SpotEye fusion is withheld until the associated paper is published."
        )

    def _gradient_magnitude(self, x):
        """Compute an internal structural descriptor (implementation withheld)."""
        raise NotImplementedError(
            "SpotEye fusion is withheld until the associated paper is published."
        )

    def _local_edge_correlation(self, rgb, depth):
        """Estimate cross-modal agreement (implementation withheld)."""
        raise NotImplementedError(
            "SpotEye fusion is withheld until the associated paper is published."
        )

    def forward(self, rgb, depth, stats):
        """Fuse RGB and depth features (implementation withheld)."""
        raise NotImplementedError(
            "SpotEye fusion is withheld until the associated paper is published."
        )


def _parse_pooler_size(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or min(int(value[0]), int(value[1])) <= 0:
            raise ValueError(f"Invalid ROI pooler size: {value!r}")
        return int(value[0]), int(value[1])
    value = int(value)
    if value <= 0:
        raise ValueError(f"Invalid ROI pooler size: {value!r}")
    return value


@META_ARCH_REGISTRY.register()
class SpotEye(ClipViper):
    """SpotEye model whose only model dependency is :class:`ClipViper`."""

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        fusion = getattr(cfg.PERSON_SEARCH.REID, "DEPTH_FUSION", None)
        dcaf = getattr(cfg.PERSON_SEARCH.REID, "DCAF", None)
        loss = getattr(cfg.PERSON_SEARCH.REID, "LOSS", None)
        depth_hn = getattr(loss, "DEPTH_HN", None) if loss is not None else None

        def read(node, key, default):
            return getattr(node, key, default) if node is not None else default

        ret.update(
            depth_fusion_norm=str(read(fusion, "NORM", "bn")),
            depth_fusion_router_hidden=int(read(fusion, "ROUTER_HIDDEN", 64)),
            depth_fusion_alpha_init_bias=float(read(fusion, "ALPHA_INIT_BIAS", 0.0)),
            depth_fusion_film_init_std=float(read(fusion, "FILM_INIT_STD", 0.01)),
            dcaf_base_ratio=float(read(dcaf, "BASE_RATIO", 0.25)),
            dcaf_aug_ratio=float(read(dcaf, "AUG_RATIO", 0.5)),
            dcaf_gate_mid=int(read(dcaf, "GATE_MID", 64)),
            dcaf_mixer_hidden=int(read(dcaf, "MIXER_HIDDEN", 32)),
            dcaf_mixer_init_bias=float(read(dcaf, "MIXER_INIT_BIAS", -1.0)),
            dcaf_aug_film_init_std=float(read(dcaf, "AUG_FILM_INIT_STD", 0.005)),
            dcaf_aug_sparsity_w=float(read(dcaf, "AUG_SPARSITY_W", 0.0)),
            dcaf_u_window_size=int(read(dcaf, "U_WINDOW_SIZE", 5)),
            dcaf_u_alpha_init=float(read(dcaf, "U_ALPHA_INIT", 4.0)),
            dcaf_u_beta_init=float(read(dcaf, "U_BETA_INIT", 0.5)),
            dcaf_u_reg_w=float(read(dcaf, "U_REG_W", 0.0)),
            dcaf_u_disp_tau=float(read(dcaf, "U_DISP_TAU", 0.05)),
            dcaf_u_usage_lo=float(read(dcaf, "U_USAGE_LO", 0.05)),
            dcaf_u_usage_hi=float(read(dcaf, "U_USAGE_HI", 0.95)),
        )
        hn_enabled = bool(read(depth_hn, "ENABLED", False))
        ret.update(
            depth_hn_weight=float(read(depth_hn, "WEIGHT", 0.0)) if hn_enabled else 0.0,
            depth_hn_alpha=float(read(depth_hn, "ALPHA", 2.0)),
            depth_hn_tau=float(read(depth_hn, "TAU", 0.07)),
            depth_preprocess_enabled=bool(read(getattr(depth_hn, "PREPROCESS", None), "ENABLED", False)),
            depth_preprocess_method=str(read(getattr(depth_hn, "PREPROCESS", None), "METHOD", "none")),
            depth_preprocess_eps=float(read(getattr(depth_hn, "PREPROCESS", None), "EPS", 1e-6)),
        )
        if hn_enabled:
            size = int(read(depth_hn, "OUTPUT_SIZE", 7))
            sampling = int(read(depth_hn, "SAMPLING_RATIO", 0))
            pooler_type = str(read(depth_hn, "POOLER_TYPE", "ROIAlignV2"))
        else:
            size = _parse_pooler_size(read(fusion, "ROI_POOLER_SIZE", 7))
            sampling = int(read(fusion, "ROI_SAMPLING_RATIO", 0))
            pooler_type = str(read(fusion, "ROI_POOLER_TYPE", "ROIAlignV2"))
        ret["depth_pooler"] = ROIPooler(
            output_size=size, scales=[1.0], sampling_ratio=sampling, pooler_type=pooler_type
        )
        return ret

    @configurable
    def __init__(
        self,
        depth_pooler: nn.Module,
        depth_preprocess_enabled=False,
        depth_preprocess_method="none",
        depth_preprocess_eps=1e-6,
        depth_hn_weight=0.0,
        depth_hn_alpha=2.0,
        depth_hn_tau=0.07,
        depth_fusion_norm="bn",
        depth_fusion_router_hidden=64,
        depth_fusion_alpha_init_bias=0.0,
        depth_fusion_film_init_std=0.01,
        dcaf_base_ratio=0.25,
        dcaf_aug_ratio=0.5,
        dcaf_gate_mid=64,
        dcaf_mixer_hidden=32,
        dcaf_mixer_init_bias=-1.0,
        dcaf_aug_film_init_std=0.005,
        dcaf_aug_sparsity_w=0.0,
        dcaf_u_window_size=5,
        dcaf_u_alpha_init=4.0,
        dcaf_u_beta_init=0.5,
        dcaf_u_reg_w=0.0,
        dcaf_u_disp_tau=0.05,
        dcaf_u_usage_lo=0.05,
        dcaf_u_usage_hi=0.95,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth_pooler = depth_pooler
        self.depth_preprocess_enabled = bool(depth_preprocess_enabled)
        self.depth_preprocess_method = str(depth_preprocess_method)
        self.depth_preprocess_eps = float(depth_preprocess_eps)
        self.depth_hn_weight = float(depth_hn_weight)
        self.depth_hn_alpha = float(depth_hn_alpha)
        self.depth_hn_tau = float(depth_hn_tau)
        self.depth_fusion_norm = str(depth_fusion_norm)
        self.depth_fusion_film_init_std = float(depth_fusion_film_init_std)
        self.dcaf_base_ratio = float(dcaf_base_ratio)
        self.dcaf_aug_ratio = float(dcaf_aug_ratio)
        self.dcaf_gate_mid = int(dcaf_gate_mid)
        self.dcaf_mixer_hidden = int(dcaf_mixer_hidden)
        self.dcaf_mixer_init_bias = float(dcaf_mixer_init_bias)
        self.dcaf_aug_film_init_std = float(dcaf_aug_film_init_std)
        self.dcaf_aug_sparsity_w = float(dcaf_aug_sparsity_w)
        self.dcaf_u_reg_w = float(dcaf_u_reg_w)
        self.dcaf_u_disp_tau = float(dcaf_u_disp_tau)
        self.dcaf_u_usage_lo = float(dcaf_u_usage_lo)
        self.dcaf_u_usage_hi = float(dcaf_u_usage_hi)
        self.deg_stats_dim = 9

        self.depth_enc3 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            _make_norm2d(64, depth_fusion_norm), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, stride=2, padding=1, bias=False),
            _make_norm2d(256, depth_fusion_norm), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            _make_norm2d(512, depth_fusion_norm), nn.ReLU(inplace=True),
        )
        self.depth_enc4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            _make_norm2d(1024, depth_fusion_norm), nn.ReLU(inplace=True),
        )
        self.depth_roi_in = nn.Conv2d(1, 2048, 1, bias=False)

        def make_fusion(channels):
            return DeterministicUFusion(
                channels, self.deg_stats_dim, depth_fusion_norm,
                dcaf_base_ratio, dcaf_aug_ratio, dcaf_gate_mid,
                dcaf_mixer_hidden, dcaf_mixer_init_bias,
                depth_fusion_film_init_std, dcaf_aug_film_init_std,
                dcaf_u_window_size, dcaf_u_alpha_init, dcaf_u_beta_init,
            )

        self.depth_adapter3 = make_fusion(512)
        self.depth_adapter4 = make_fusion(1024)
        self.depth_roi_adapter = make_fusion(2048)
        self.router3 = AlphaRouter(self.deg_stats_dim, depth_fusion_router_hidden, depth_fusion_alpha_init_bias)
        self.router4 = AlphaRouter(self.deg_stats_dim, depth_fusion_router_hidden, depth_fusion_alpha_init_bias)
        self.router_roi = AlphaRouter(self.deg_stats_dim, depth_fusion_router_hidden, depth_fusion_alpha_init_bias)
        self._depth_alpha_dict: Dict[str, torch.Tensor] = {}
        self._dcaf_info: Dict[str, dict] = {}

    def forward(self, input_list, infgt=False):
        if "query" in input_list[0]:
            images = self.preprocess_input([item["query"] for item in input_list])
            targets = [item["query"]["instances"].to(self.device) for item in input_list]
            return self.forward_query(images, targets)
        images = self.preprocess_input(input_list)
        targets = [item["instances"].to(self.device) for item in input_list]
        depths = self.preprocess_depth([item["depth"] for item in input_list]) if "depth" in input_list[0] else None
        if self.training:
            predictions, features, losses = self.forward_gallery(images, depths, targets)
            self.visualize_training(images, targets, predictions, features)
            return losses
        if infgt:
            return self.forward_gallery_gt(images, targets)
        return self.forward_gallery(images, depths, targets)

    def preprocess_depth(self, tensors):
        normalized = []
        for depth in tensors:
            depth = depth.to(self.device).float()
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 3 and depth.shape[0] == 3:
                depth = depth.mean(0, keepdim=True)
            if depth.max() > 1.5:
                depth = depth / 255.0
            if self.depth_preprocess_enabled and self.depth_preprocess_method == "log_median":
                valid = torch.isfinite(depth) & (depth > 0)
                if valid.any():
                    depth = depth / (depth[valid].median() + self.depth_preprocess_eps)
                depth = torch.log(depth + self.depth_preprocess_eps)
            normalized.append(depth)
        return ImageList.from_tensors(normalized, getattr(self, "size_divisibility", 32))

    def _stats(self, rgb, depth):
        return torch.cat([_depth_stats(depth), _rgb_stats(rgb)], dim=1)

    def _fuse(self, rgb, depth, stats, adapter, router, tag):
        """Apply SpotEye feature fusion at one backbone stage.

        The complete implementation will be released after paper publication.
        """
        raise NotImplementedError(
            "SpotEye backbone fusion is withheld until the paper is published."
        )

    def backbone(self, x, depth_list=None):
        """Extract and fuse backbone features (implementation withheld).

        The complete implementation will be released after paper publication.
        """
        raise NotImplementedError(
            "SpotEye backbone is withheld until the associated paper is published."
        )

    def _collect_patches(self, instances, mask, image_list):
        if image_list is None or not mask.any():
            return None
        boxes, cursor = [], 0
        for instance in instances:
            count = len(instance.pred_boxes)
            boxes.append(instance.pred_boxes[mask[cursor:cursor + count]])
            cursor += count
        return self.depth_pooler([image_list.tensor], boxes)

    def _fuse_roi(self, box_features, depth_patches, rgb_patches):
        """Fuse region-level RGB and depth features (implementation withheld).

        The complete implementation will be released after paper publication.
        """
        raise NotImplementedError(
            "SpotEye ROI fusion is withheld until the associated paper is published."
        )

    def forward_gallery(self, image_list, depth_list, gt_instances):
        self._dcaf_info = {}
        self._depth_alpha_dict = {}
        if not self.training:
            # Preserve the established SpotEye inference contract: backbone fusion
            # is active, while post-ROI fusion is a training-only branch.
            return ClipViper.forward_gallery(self, image_list, gt_instances, depth_list=depth_list)

        features = self.backbone(image_list.tensor, depth_list)
        proposals, proposal_losses = self.proposal_generator(image_list, features, gt_instances)
        pred_instances, box_features, losses, matches = self.roi_heads(
            image_list, features, proposals, gt_instances
        )
        losses.update(proposal_losses)
        pseudo_id = 1e5
        pseudo_targets = []
        for instance in gt_instances:
            ids = copy.deepcopy(instance.gt_pids)
            for index, pid in enumerate(ids):
                if pid == -1:
                    ids[index], pseudo_id = pseudo_id, pseudo_id + 1
            pseudo_targets.append(ids)
        ids_per_image = self.id_assigner(
            [x.pred_boxes.tensor for x in pred_instances],
            [x.pred_scores for x in pred_instances],
            [x.gt_boxes.tensor for x in gt_instances],
            pseudo_targets,
            match_indices=matches,
        )
        ids = torch.cat(ids_per_image)
        positive = ids > -2
        box_features, ids = box_features[positive], ids[positive]
        hn_meta = {"camera_ids": None, "image_ids": None, "bbox_areas": None}
        if getattr(self.hard_negative_loss, "enabled", False):
            hn_meta = build_roi_queue_metadata(
                pred_instances, ids_per_image, gt_instances, positive, self.device
            )
        depth_patches = self._collect_patches(pred_instances, positive, depth_list)
        rgb_patches = self._collect_patches(pred_instances, positive, image_list)
        box_features = self._fuse_roi(box_features, depth_patches, rgb_patches)
        depth_sig = _depth_signature(depth_patches) if depth_patches is not None else None
        box_embs, ids = self.img_embes(box_features, ids)
        box_tokens = box_features.flatten(2, 3).permute(2, 0, 1).contiguous()
        box_embs = self.bn_neck(box_embs)
        ids[ids >= 1e5] = -1
        for name, value in self.oim_loss(box_embs, ids).items():
            losses[name] = value * 0.5
        if depth_sig is not None and self.depth_hn_weight > 0:
            if len(box_embs) % len(depth_sig) == 0:
                expanded_sig = depth_sig.repeat(len(box_embs) // len(depth_sig), 1)
                losses["loss_depth_hn"] = self.depth_hn_weight * _depth_aware_supcon(
                    box_embs, ids, expanded_sig, self.depth_hn_alpha, self.depth_hn_tau
                )

        image_features, dense_tokens, text_tokens, text_ids = [], [], [], []
        counts = [(x > -2).nonzero().shape[0] for x in ids_per_image]
        roi_ids = [x[x > -2] for x in ids_per_image]
        roi_features = torch.split(box_tokens.permute(1, 0, 2).contiguous(), counts)
        for image_rois, image_ids, target in zip(roi_features, roi_ids, gt_instances):
            for pid, descriptions in zip(target.gt_pids, target.descriptions):
                if pid > -1:
                    matched = image_rois[image_ids == pid]
                    image_features.append(matched.unsqueeze(1).expand(-1, len(descriptions), -1, -1).flatten(0, 1))
                    dense_tokens.extend([token for _ in range(len(matched)) for token in descriptions])
                    text_tokens.extend(descriptions)
                    text_ids.extend([pid] * len(descriptions))
        if image_features:
            losses.update(self.mim_loss(torch.cat(image_features), dense_tokens))
        else:
            losses["loss_mim"] = box_embs.sum() * 0.0
        if text_ids:
            text_ids = torch.stack(text_ids).to(self.device)
            text_embs = self.text_embeds(torch.stack(text_tokens).to(self.device))
            for name, value in self.oim_loss(text_embs, text_ids).items():
                losses[name + "_text"] = value * 0.5
            labelled = ids > -1
            losses["loss_sdm"] = self.compute_sdm(
                box_embs[labelled], text_embs, ids[labelled], text_ids, 1 / 0.02
            )
            if getattr(self.hard_negative_loss, "enabled", False):
                hn_meta = expand_queue_metadata(hn_meta, int(positive.sum()), box_embs.shape[0])
                add_hard_negative_queue_loss(
                    self.hard_negative_loss, losses, text_embs, text_ids, box_embs, ids,
                    camera_ids=hn_meta["camera_ids"], image_ids=hn_meta["image_ids"],
                    bbox_areas=hn_meta["bbox_areas"],
                )
        else:
            losses["loss_oim_text"] = box_embs.sum() * 0.0
            losses["loss_sdm"] = box_embs.sum() * 0.0
            if getattr(self.hard_negative_loss, "enabled", False):
                losses["loss_hard_negative"] = box_embs.sum() * 0.0
        for index, instance in enumerate(pred_instances):
            assigned = ids_per_image[index]
            assigned[assigned >= 1e5] = -1
            instance.assign_ids = assigned
        self._add_auxiliary_losses(losses)
        self._log_spoteye_stats()
        return pred_instances, [value.detach() for value in features.values()], losses

    def _add_auxiliary_losses(self, losses):
        gates = [info["U"] for info in self._dcaf_info.values() if "U" in info]
        if self.dcaf_aug_sparsity_w > 0 and gates:
            terms = []
            for info in self._dcaf_info.values():
                terms.append((info["U"] * info["phi_aug"].abs().mean(1, keepdim=True)).mean())
            losses["loss_dcaf_sparsity"] = self.dcaf_aug_sparsity_w * torch.stack(terms).mean()
        if self.dcaf_u_reg_w > 0 and gates:
            dispersion = [F.relu(self.dcaf_u_disp_tau - gate.float().flatten(1).std(1)).mean() for gate in gates]
            usage = [
                F.relu(self.dcaf_u_usage_lo - gate.float().mean())
                + F.relu(gate.float().mean() - self.dcaf_u_usage_hi)
                for gate in gates
            ]
            losses["loss_dcaf_u_reg"] = self.dcaf_u_reg_w * (
                torch.stack(dispersion).mean() + torch.stack(usage).mean()
            )

    def _log_spoteye_stats(self):
        if not self._dcaf_info:
            return
        try:
            storage = get_event_storage()
        except Exception:
            return
        for tag, info in self._dcaf_info.items():
            gate = info["U"].detach()
            storage.put_scalar(f"dcaf/{tag}/U_mean", float(gate.mean()))
            storage.put_scalar(f"dcaf/{tag}/U_std", float(gate.std()))
            storage.put_scalar(
                f"dcaf/{tag}/phi_aug_norm",
                float(info["phi_aug"].detach().square().mean().sqrt()),
            )
