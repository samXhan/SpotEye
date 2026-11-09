import unittest

import torch
from torchvision.ops import roi_align

from psd2.layers import ROIAlign


class TestNPURoiAlign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_npu
        except ModuleNotFoundError as error:
            if error.name != "torch_npu":
                raise
            raise unittest.SkipTest("TorchNPU is required") from error
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("An Ascend NPU is required")
        cls.device = torch.device("npu", torch_npu.npu.current_device())

    def test_rectangular_pooling_matches_torchvision(self):
        generator = torch.Generator().manual_seed(9)
        features = torch.randn(2, 32, 23, 19, generator=generator)
        rois = torch.tensor([
            [0, 0, 0, 304, 368],
            [0, 53.25, 20.5, 229.75, 332.25],
            [1, 4.25, 2.5, 5.75, 3.75],
            [1, 32.5, 40, 189, 257.25],
        ])
        grad = torch.randn(4, 32, 16, 8, generator=generator)
        for aligned, sampling_ratio in ((True, 0), (True, 2), (False, 0)):
            with self.subTest(aligned=aligned, sampling_ratio=sampling_ratio):
                cpu = features.clone().requires_grad_()
                npu = features.to(self.device).requires_grad_()
                expected = roi_align(cpu, rois, (16, 8), 1 / 16, sampling_ratio, aligned)
                actual = ROIAlign((16, 8), 1 / 16, sampling_ratio, aligned)(
                    npu, rois.to(self.device)
                )
                self.assertEqual(actual.device, self.device)
                expected.backward(grad)
                actual.backward(grad.to(self.device))
                self.assertEqual(npu.grad.device, self.device)
                torch.testing.assert_close(actual.detach().cpu(), expected.detach(),
                                           rtol=2e-3, atol=3e-4)
                torch.testing.assert_close(npu.grad.cpu(), cpu.grad, rtol=2e-3, atol=3e-4)

    def test_empty_rois_keep_input_gradient(self):
        features = torch.ones(2, 32, 4, 5, device=self.device, requires_grad=True)
        rois = features.new_empty((0, 5))
        output = ROIAlign((16, 8), 1 / 16, 0)(features, rois)
        self.assertEqual(output.shape, (0, 32, 16, 8))
        self.assertEqual(output.device, self.device)
        output.sum().backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))
