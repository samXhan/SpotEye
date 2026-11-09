# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align


class _NPUROIAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, rois, height, width, scale, sampling_ratio, end_mode):
        ctx.save_for_backward(rois)
        ctx.input_shape = features.shape
        ctx.output_size = height, width
        ctx.scale = scale
        ctx.sampling_ratio = sampling_ratio
        ctx.end_mode = end_mode
        if rois.numel() == 0:
            return features.new_empty((0, features.shape[1], height, width))
        return torch.ops.npu.npu_roi_align(
            features, rois, scale, height, width, sampling_ratio, end_mode
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        height, width = ctx.output_size
        if rois.numel() == 0:
            grad_input = grad_output.new_zeros(ctx.input_shape)
        else:
            grad_input = torch.ops.npu.npu_roi_alignbk(
                grad_output.contiguous(),
                rois,
                ctx.input_shape,
                width,
                height,
                ctx.scale,
                ctx.sampling_ratio,
                ctx.end_mode,
            )
        return grad_input, None, None, None, None, None, None


# NOTE: torchvision's RoIAlign has a different default aligned=False
class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        from torchvision import __version__

        version = tuple(int(x) for x in __version__.split(".")[:2])
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if input.is_quantized:
            input = input.dequantize()
        if input.device.type == "npu":
            return self._forward_npu(input, rois)
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    @torch.jit.unused
    def _forward_npu(self, input, rois):
        height, width = _pair(self.output_size)
        pooled = _NPUROIAlign.apply(
            input.float(),
            rois.to(device=input.device, dtype=torch.float32).contiguous(),
            height,
            width,
            self.spatial_scale,
            max(self.sampling_ratio, 0),
            2 if self.aligned else 0,
        )
        return pooled.to(dtype=input.dtype)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
