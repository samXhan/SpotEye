import re
import zlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_loss_like(*tensors: torch.Tensor) -> torch.Tensor:
    for tensor in tensors:
        if tensor is not None and torch.is_tensor(tensor):
            return tensor.sum() * 0.0
    return torch.tensor(0.0)


def _safe_storage():
    try:
        from psd2.utils.events import get_event_storage

        return get_event_storage()
    except Exception:
        return None


def _put_scalar(storage, name: str, value) -> None:
    if storage is None:
        return
    try:
        if torch.is_tensor(value):
            value = value.detach().float().item()
        storage.put_scalar(name, float(value))
    except Exception:
        return


def _stable_image_hash(image_id) -> int:
    if image_id is None:
        return -1
    return int(zlib.crc32(str(image_id).encode("utf-8")) & 0x7FFFFFFF)


def _parse_camera_id(image_id) -> int:
    if image_id is None:
        return -1
    # PRW frame names are typically "c1s2_071096.jpg".
    match = re.search(r"(?:^|[/\\])c(\d+)s\d+_", str(image_id))
    if match is None:
        match = re.search(r"^c(\d+)s\d+_", str(image_id))
    return int(match.group(1)) if match is not None else -1


def build_roi_queue_metadata(
    pred_instances,
    assign_ids_per_img,
    gt_instances,
    pos_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    """Build ROI-aligned optional metadata for hard-negative diagnostics.

    The returned tensors are aligned with the pre-embedding ROI rows after
    filtering by ``pos_mask``. Missing fields are represented as valid fallback
    tensors (-1 for ids, box area from predicted boxes when available).
    """
    if len(assign_ids_per_img) == 0:
        return {"camera_ids": None, "image_ids": None, "bbox_areas": None}

    counts = [int(ids.numel()) for ids in assign_ids_per_img]
    if sum(counts) != int(pos_mask.numel()):
        return {"camera_ids": None, "image_ids": None, "bbox_areas": None}

    camera_chunks: List[torch.Tensor] = []
    image_chunks: List[torch.Tensor] = []
    for count, inst in zip(counts, gt_instances):
        image_id = inst.image_id if hasattr(inst, "has") and inst.has("image_id") else None
        camera_value = _parse_camera_id(image_id)
        image_value = _stable_image_hash(image_id)
        camera_chunks.append(torch.full((count,), camera_value, dtype=torch.long, device=device))
        image_chunks.append(torch.full((count,), image_value, dtype=torch.long, device=device))

    camera_ids = torch.cat(camera_chunks, dim=0)[pos_mask]
    image_ids = torch.cat(image_chunks, dim=0)[pos_mask]

    bbox_areas = None
    try:
        all_boxes = torch.cat([inst.pred_boxes.tensor for inst in pred_instances], dim=0).to(device)
        if all_boxes.shape[0] == pos_mask.numel():
            boxes = all_boxes[pos_mask]
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
            bbox_areas = (wh[:, 0] * wh[:, 1]).float()
    except Exception:
        bbox_areas = None

    return {"camera_ids": camera_ids, "image_ids": image_ids, "bbox_areas": bbox_areas}


def expand_queue_metadata(
    metadata: Dict[str, Optional[torch.Tensor]],
    base_size: int,
    expanded_size: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """Repeat ROI metadata when the embedding branch expands each ROI row."""
    if base_size <= 0 or expanded_size == base_size:
        return metadata
    if expanded_size % base_size != 0:
        return {key: None for key in metadata}
    rep = expanded_size // base_size
    expanded = {}
    for key, value in metadata.items():
        expanded[key] = value.repeat(rep) if value is not None else None
    return expanded


def _align_text_to_roi(
    text_feats: torch.Tensor,
    text_pids: torch.Tensor,
    roi_feats: torch.Tensor,
    roi_pids: torch.Tensor,
    camera_ids: Optional[torch.Tensor] = None,
    image_ids: Optional[torch.Tensor] = None,
    bbox_areas: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if text_feats.numel() == 0 or roi_feats.numel() == 0:
        empty_text = text_feats[:0]
        empty_roi = roi_feats[:0]
        empty_pid = text_pids[:0].to(dtype=torch.long)
        return empty_text, empty_roi, empty_pid, None, None, None

    text_pids = text_pids.to(device=text_feats.device, dtype=torch.long)
    roi_pids = roi_pids.to(device=roi_feats.device, dtype=torch.long)
    valid_text = text_pids > -1
    valid_roi = roi_pids > -1
    if valid_text.sum() == 0 or valid_roi.sum() == 0:
        empty_text = text_feats[:0]
        empty_roi = roi_feats[:0]
        empty_pid = text_pids[:0]
        return empty_text, empty_roi, empty_pid, None, None, None

    t = text_feats[valid_text]
    t_pid = text_pids[valid_text]
    r = roi_feats[valid_roi]
    r_pid = roi_pids[valid_roi]
    valid_roi_indices = valid_roi.nonzero(as_tuple=False).flatten()

    with torch.no_grad():
        sim = F.normalize(t.float(), dim=-1) @ F.normalize(r.float(), dim=-1).t()
        same_pid = t_pid.view(-1, 1) == r_pid.view(1, -1)
        sim = sim.masked_fill(~same_pid, float("-inf"))
        best_sim, best_local = sim.max(dim=1)
        keep = torch.isfinite(best_sim)

    if keep.sum() == 0:
        empty_text = text_feats[:0]
        empty_roi = roi_feats[:0]
        empty_pid = text_pids[:0]
        return empty_text, empty_roi, empty_pid, None, None, None

    text_keep = valid_text.nonzero(as_tuple=False).flatten()[keep]
    roi_keep = valid_roi_indices[best_local[keep]]
    pair_text = text_feats[text_keep]
    pair_roi = roi_feats[roi_keep]
    pair_pids = text_pids[text_keep]

    def _select_optional(values):
        if values is None:
            return None
        values = values.to(device=roi_feats.device)
        if values.numel() != roi_pids.numel():
            return None
        return values[roi_keep]

    pair_camera = _select_optional(camera_ids)
    pair_image = _select_optional(image_ids)
    pair_area = _select_optional(bbox_areas)
    return pair_text, pair_roi, pair_pids, pair_camera, pair_image, pair_area


def add_hard_negative_queue_loss(
    loss_module: Optional[nn.Module],
    losses: Dict[str, torch.Tensor],
    text_feats: torch.Tensor,
    text_pids: torch.Tensor,
    roi_feats: torch.Tensor,
    roi_pids: torch.Tensor,
    camera_ids: Optional[torch.Tensor] = None,
    image_ids: Optional[torch.Tensor] = None,
    bbox_areas: Optional[torch.Tensor] = None,
) -> None:
    if loss_module is None or not getattr(loss_module, "enabled", False):
        return
    pair_text, pair_roi, pair_pids, pair_camera, pair_image, pair_area = _align_text_to_roi(
        text_feats=text_feats,
        text_pids=text_pids,
        roi_feats=roi_feats,
        roi_pids=roi_pids,
        camera_ids=camera_ids,
        image_ids=image_ids,
        bbox_areas=bbox_areas,
    )
    losses["loss_hard_negative"] = loss_module(
        text_feats=pair_text,
        roi_feats=pair_roi,
        person_ids=pair_pids,
        camera_ids=pair_camera,
        image_ids=pair_image,
        bbox_areas=pair_area,
    )


class HardNegativeQueueLoss(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        enabled: bool = False,
        queue_size: int = 8192,
        topk: int = 32,
        temperature: float = 0.07,
        loss_weight: float = 0.1,
        warmup_iters: int = 1000,
        same_camera_weight: float = 1.2,
        area_close_weight: float = 1.1,
        area_close_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.feat_dim = int(feat_dim)
        self.queue_size = int(queue_size)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.loss_weight = float(loss_weight)
        self.warmup_iters = int(warmup_iters)
        self.same_camera_weight = float(same_camera_weight)
        self.area_close_weight = float(area_close_weight)
        self.area_close_ratio = float(area_close_ratio)

        self.register_buffer("roi_queue", torch.zeros(self.queue_size, self.feat_dim))
        self.register_buffer("text_queue", torch.zeros(self.queue_size, self.feat_dim))
        self.register_buffer("pid_queue", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("camera_queue", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("image_queue", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("area_queue", torch.full((self.queue_size,), -1.0, dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.PERSON_SEARCH.REID.LOSS
        hn = getattr(loss_cfg, "HARD_NEGATIVE", None)
        oim = getattr(loss_cfg, "OIM", None)
        feat_dim = int(getattr(oim, "FEAT_DIM", cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM))
        return {
            "feat_dim": feat_dim,
            "enabled": bool(getattr(hn, "ENABLED", False)) if hn is not None else False,
            "queue_size": int(getattr(hn, "QUEUE_SIZE", 8192)) if hn is not None else 8192,
            "topk": int(getattr(hn, "TOPK", 32)) if hn is not None else 32,
            "temperature": float(getattr(hn, "TEMPERATURE", 0.07)) if hn is not None else 0.07,
            "loss_weight": float(getattr(hn, "LOSS_WEIGHT", 0.1)) if hn is not None else 0.1,
            "warmup_iters": int(getattr(hn, "WARMUP_ITERS", 1000)) if hn is not None else 1000,
            "same_camera_weight": float(getattr(hn, "SAME_CAMERA_WEIGHT", 1.2)) if hn is not None else 1.2,
            "area_close_weight": float(getattr(hn, "AREA_CLOSE_WEIGHT", 1.1)) if hn is not None else 1.1,
            "area_close_ratio": float(getattr(hn, "AREA_CLOSE_RATIO", 0.3)) if hn is not None else 0.3,
        }

    @property
    def current_size(self) -> int:
        return int(self.queue_count.item())

    def _current_iter(self) -> int:
        storage = _safe_storage()
        if storage is not None:
            try:
                return int(storage.iter)
            except Exception:
                pass
        return int(self.num_updates.item())

    def _queue_view(self):
        count = self.current_size
        return (
            self.roi_queue[:count],
            self.text_queue[:count],
            self.pid_queue[:count],
            self.camera_queue[:count],
            self.image_queue[:count],
            self.area_queue[:count],
        )

    @torch.no_grad()
    def _enqueue(
        self,
        text_feats: torch.Tensor,
        roi_feats: torch.Tensor,
        person_ids: torch.Tensor,
        camera_ids: Optional[torch.Tensor],
        image_ids: Optional[torch.Tensor],
        bbox_areas: Optional[torch.Tensor],
    ) -> None:
        if self.queue_size <= 0 or text_feats.numel() == 0 or roi_feats.numel() == 0:
            self.num_updates += 1
            return
        valid = person_ids.to(device=roi_feats.device, dtype=torch.long) > -1
        valid = valid & torch.isfinite(text_feats).all(dim=1) & torch.isfinite(roi_feats).all(dim=1)
        if valid.sum() == 0:
            self.num_updates += 1
            return

        text = F.normalize(text_feats[valid].detach().float(), dim=-1)
        roi = F.normalize(roi_feats[valid].detach().float(), dim=-1)
        pids = person_ids[valid].detach().to(device=roi_feats.device, dtype=torch.long)
        n = int(pids.numel())
        if n > self.queue_size:
            text = text[-self.queue_size :]
            roi = roi[-self.queue_size :]
            pids = pids[-self.queue_size :]
            valid_idx = valid.nonzero(as_tuple=False).flatten()[-self.queue_size :]
            n = self.queue_size
        else:
            valid_idx = valid.nonzero(as_tuple=False).flatten()

        def _optional_or_fill(values, fill, dtype):
            if values is None or values.numel() != valid.numel():
                return torch.full((n,), fill, dtype=dtype, device=roi_feats.device)
            return values.to(device=roi_feats.device, dtype=dtype)[valid_idx]

        cams = _optional_or_fill(camera_ids, -1, torch.long)
        imgs = _optional_or_fill(image_ids, -1, torch.long)
        areas = _optional_or_fill(bbox_areas, -1.0, torch.float32)

        ptr = int(self.queue_ptr.item())
        end = ptr + n
        if end <= self.queue_size:
            sl = slice(ptr, end)
            self.roi_queue[sl] = roi
            self.text_queue[sl] = text
            self.pid_queue[sl] = pids
            self.camera_queue[sl] = cams
            self.image_queue[sl] = imgs
            self.area_queue[sl] = areas
        else:
            first = self.queue_size - ptr
            second = n - first
            self.roi_queue[ptr:] = roi[:first]
            self.text_queue[ptr:] = text[:first]
            self.pid_queue[ptr:] = pids[:first]
            self.camera_queue[ptr:] = cams[:first]
            self.image_queue[ptr:] = imgs[:first]
            self.area_queue[ptr:] = areas[:first]
            self.roi_queue[:second] = roi[first:]
            self.text_queue[:second] = text[first:]
            self.pid_queue[:second] = pids[first:]
            self.camera_queue[:second] = cams[first:]
            self.image_queue[:second] = imgs[first:]
            self.area_queue[:second] = areas[first:]

        self.queue_ptr[0] = (ptr + n) % self.queue_size
        self.queue_count[0] = min(self.queue_size, int(self.queue_count.item()) + n)
        self.num_updates += 1

    def _log(
        self,
        loss_raw: torch.Tensor,
        loss_weighted: torch.Tensor,
        valid_pairs: int,
        avg_neg_sim: float,
        avg_pos_sim: float,
        same_camera_ratio: float,
        effective_topk: float,
    ) -> None:
        storage = _safe_storage()
        _put_scalar(storage, "hn/loss_raw", loss_raw)
        _put_scalar(storage, "hn/loss_weighted", loss_weighted)
        _put_scalar(storage, "hn/queue_size", self.current_size)
        _put_scalar(storage, "hn/valid_pairs", valid_pairs)
        _put_scalar(storage, "hn/avg_neg_sim", avg_neg_sim)
        _put_scalar(storage, "hn/avg_pos_sim", avg_pos_sim)
        _put_scalar(storage, "hn/pos_neg_margin", avg_pos_sim - avg_neg_sim)
        _put_scalar(storage, "hn/same_camera_neg_ratio", same_camera_ratio)
        _put_scalar(storage, "hn/effective_topk", effective_topk)

    def forward(
        self,
        text_feats: torch.Tensor,
        roi_feats: torch.Tensor,
        person_ids: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        bbox_areas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return _zero_loss_like(text_feats, roi_feats)

        if text_feats.shape[0] != roi_feats.shape[0] or text_feats.shape[0] != person_ids.shape[0]:
            zero = _zero_loss_like(text_feats, roi_feats)
            self._enqueue(text_feats, roi_feats, person_ids, camera_ids, image_ids, bbox_areas)
            self._log(zero, zero, 0, 0.0, 0.0, 0.0, 0.0)
            return zero

        text = F.normalize(text_feats.float(), dim=-1)
        roi = F.normalize(roi_feats.float(), dim=-1)
        pids = person_ids.to(device=roi_feats.device, dtype=torch.long)

        q_roi, _, q_pids, q_cams, _, q_areas = self._queue_view()
        q_roi = q_roi.clone().detach()
        q_pids = q_pids.clone().detach()
        q_cams = q_cams.clone().detach()
        q_areas = q_areas.clone().detach()
        zero = _zero_loss_like(text_feats, roi_feats)
        if self.current_size == 0 or text.numel() == 0:
            self._enqueue(text_feats, roi_feats, pids, camera_ids, image_ids, bbox_areas)
            self._log(zero, zero, 0, 0.0, 0.0, 0.0, 0.0)
            return zero

        valid_current = pids > -1
        valid_current = valid_current & torch.isfinite(text).all(dim=1) & torch.isfinite(roi).all(dim=1)
        if valid_current.sum() == 0:
            self._enqueue(text_feats, roi_feats, pids, camera_ids, image_ids, bbox_areas)
            self._log(zero, zero, 0, 0.0, 0.0, 0.0, 0.0)
            return zero

        text_v = text[valid_current]
        roi_v = roi[valid_current]
        pids_v = pids[valid_current]
        pos_sim = (text_v * roi_v).sum(dim=1)

        q_roi = q_roi.to(device=text_v.device, dtype=text_v.dtype)
        q_pids = q_pids.to(device=text_v.device)
        sims = text_v @ q_roi.t()
        neg_mask = (q_pids.view(1, -1) > -1) & (q_pids.view(1, -1) != pids_v.view(-1, 1))
        if neg_mask.sum() == 0:
            self._enqueue(text_feats, roi_feats, pids, camera_ids, image_ids, bbox_areas)
            self._log(zero, zero, 0, 0.0, float(pos_sim.mean().detach().item()), 0.0, 0.0)
            return zero

        masked_sims = sims.masked_fill(~neg_mask, float("-inf"))
        k = min(max(int(self.topk), 1), masked_sims.shape[1])
        top_vals, top_idx = torch.topk(masked_sims, k=k, dim=1)
        top_valid = torch.isfinite(top_vals)
        row_valid = top_valid.any(dim=1)
        if row_valid.sum() == 0:
            self._enqueue(text_feats, roi_feats, pids, camera_ids, image_ids, bbox_areas)
            self._log(zero, zero, 0, 0.0, float(pos_sim.mean().detach().item()), 0.0, 0.0)
            return zero

        weights = torch.ones_like(top_vals)
        same_camera_ratio = 0.0
        if camera_ids is not None and camera_ids.numel() == pids.numel():
            cam_v = camera_ids.to(device=text_v.device, dtype=torch.long)[valid_current]
            q_cams = q_cams.to(device=text_v.device)
            neg_cams = q_cams[top_idx]
            same_cam = (cam_v.view(-1, 1) > -1) & (cam_v.view(-1, 1) == neg_cams)
            same_cam = same_cam & top_valid
            weights = torch.where(same_cam, weights * self.same_camera_weight, weights)
            denom = top_valid.float().sum().clamp_min(1.0)
            same_camera_ratio = float(same_cam.float().sum().detach().item() / denom.detach().item())

        if bbox_areas is not None and bbox_areas.numel() == pids.numel():
            area_v = bbox_areas.to(device=text_v.device, dtype=torch.float32)[valid_current]
            q_areas = q_areas.to(device=text_v.device, dtype=torch.float32)
            neg_areas = q_areas[top_idx]
            area_ratio = (neg_areas / area_v.clamp_min(1.0).view(-1, 1) - 1.0).abs()
            close_area = (area_v.view(-1, 1) > 0) & (neg_areas > 0) & (area_ratio <= self.area_close_ratio)
            close_area = close_area & top_valid
            weights = torch.where(close_area, weights * self.area_close_weight, weights)

        neg_logits = top_vals / self.temperature
        neg_logits = neg_logits + weights.clamp_min(1.0e-6).log()
        neg_logits = neg_logits.masked_fill(~top_valid, float("-inf"))
        pos_logits = (pos_sim / self.temperature).view(-1, 1)
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        losses = torch.logsumexp(all_logits, dim=1) - pos_logits.squeeze(1)
        losses = losses[row_valid]
        loss_raw = losses.mean() if losses.numel() > 0 else zero
        active = self._current_iter() >= self.warmup_iters
        loss_weighted = loss_raw * self.loss_weight if active else loss_raw * 0.0

        valid_top = top_valid[row_valid]
        neg_vals = top_vals[row_valid][valid_top]
        avg_neg_sim = float(neg_vals.mean().detach().item()) if neg_vals.numel() > 0 else 0.0
        avg_pos_sim = float(pos_sim[row_valid].mean().detach().item()) if row_valid.sum() > 0 else 0.0
        effective_topk = float(valid_top.float().sum(dim=1).mean().detach().item()) if valid_top.numel() > 0 else 0.0

        self._enqueue(text_feats, roi_feats, pids, camera_ids, image_ids, bbox_areas)
        self._log(
            loss_raw=loss_raw,
            loss_weighted=loss_weighted,
            valid_pairs=int(row_valid.sum().detach().item()),
            avg_neg_sim=avg_neg_sim,
            avg_pos_sim=avg_pos_sim,
            same_camera_ratio=same_camera_ratio,
            effective_topk=effective_topk,
        )
        return loss_weighted
