"""Checkpoint compatibility and FP32 inference reference smoke checks."""
import unittest
import torch
from support import ROOT, CUDATestCase


class PretrainedFP32(CUDATestCase):
    def test_dncnn_and_srresnet_checkpoints(self):
        from models.util_converse import Converse2D
        from models.converse_dncnn import ConverseDnCNN
        from models.converse_srresnet import ConverseMSRResNet
        torch.manual_seed(17)
        for name, factory, channels in (
            ('converse_dncnn', ConverseDnCNN, 1),
            ('converse_srresnet', ConverseMSRResNet, 3),
        ):
            with self.subTest(model=name):
                model = factory().cuda().eval()
                model.load_state_dict(torch.load(ROOT/'model_zoo'/f'{name}.pth', map_location='cuda', weights_only=True), strict=True)
                x = torch.rand(1, channels, 24, 32, device='cuda')
                with torch.inference_mode():
                    for layer in model.modules():
                        if isinstance(layer, Converse2D):
                            layer.backend = 'pytorch'
                    reference = model(x)
                    for layer in model.modules():
                        if isinstance(layer, Converse2D):
                            layer.backend = 'cuda'
                    output = model(x)
                torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())
                torch.ops.converse2d.clear_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)
