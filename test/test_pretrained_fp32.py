"""Checkpoint compatibility and FP32 inference reference smoke checks."""
import unittest
import copy
from contextlib import contextmanager
from unittest.mock import patch
import torch
from support import ROOT, CUDATestCase
from numerical_policy import error_metrics, write_report
from fp32_baseline import converse2d_reference as baseline_reference


@contextmanager
def frozen_model_reference():
    with patch('models.util_converse.converse2d_reference', baseline_reference), \
         patch('models.converse_usrnet.converse2d_reference', baseline_reference):
        yield


def set_backend(model, backend):
    for layer in model.modules():
        if hasattr(layer, 'backend'):
            layer.backend = backend


def model_comparison(actual, expected, *, atol=1e-5, rtol=1e-5):
    row = error_metrics(actual, expected.double())
    row['passed'] = row['finite'] and bool(torch.isclose(actual, expected, atol=atol, rtol=rtol).all())
    row.update(atol=atol, rtol=rtol)
    if actual.numel() == 1 and row['finite']:
        row.update(candidate_value=actual.item(), baseline_value=expected.item())
    return row


class PretrainedFP32(CUDATestCase):
    def setUp(self):
        # A process-wide backend override must not turn both runs into CUDA.
        override = patch.dict('os.environ', {'CONVERSE2D_BACKEND': ''})
        override.start()
        self.addCleanup(override.stop)

    def test_pretrained_model_outputs(self):
        from models.converse_dncnn import ConverseDnCNN
        from models.converse_srresnet import ConverseMSRResNet
        from models.converse_usrnet import ConverseUSRNet
        torch.manual_seed(17)
        rows = []
        self.addCleanup(lambda: print('Model report:', write_report('pretrained', rows, device='cuda',
            extra={'complete': len(rows) == 6, 'reference': 'frozen full-spectrum Python FP32'})))
        for name, factory, channels in (
            ('converse_dncnn', ConverseDnCNN, 1),
            ('converse_srresnet', ConverseMSRResNet, 3),
            ('converse_usrnet', ConverseUSRNet, 3),
        ):
            with self.subTest(model=name):
                model = factory().cuda().eval()
                model.load_state_dict(torch.load(ROOT/'model_zoo'/f'{name}.pth', map_location='cuda', weights_only=True), strict=True)
                x = torch.rand(1, channels, 24, 32, device='cuda')
                kernel = torch.softmax(torch.randn(1, 1, 49, device='cuda'), -1).reshape(1, 1, 7, 7)
                for scale in ((1, 2, 3, 4) if name == 'converse_usrnet' else (None,)):
                    args = (x, kernel, scale) if scale is not None else (x,)
                    with torch.inference_mode():
                        set_backend(model, 'pytorch')
                        with frozen_model_reference():
                            reference = model(*args)
                        set_backend(model, 'cuda')
                        output = model(*args)
                    row = dict(model=name, scale=scale, **model_comparison(output, reference))
                    rows.append(row)
                    self.assertTrue(row['passed'], row)
                    self.assertEqual(output.dtype, torch.float32)
                torch.ops.converse2d.clear_cache()

    def test_usrnet_three_seed_three_step_training_regression(self):
        """Small real USRNet/Adam trajectories; explicitly not convergence proof."""
        from models.converse_usrnet import ConverseUSRNet
        rows = []
        self.addCleanup(lambda: print('Training report:', write_report('usrnet_training', rows, device='cuda',
            extra={'seeds': [17, 29, 43], 'steps': 3, 'num_iterations': 2, 'num_blocks': 1,
                   'reference': 'frozen full-spectrum Python FP32',
                   'complete': sum(row['tensor'] == 'loss' for row in rows) == 9,
                   'convergence_evidence': False})))
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(lambda: torch.use_deterministic_algorithms(deterministic, warn_only=warn_only))
        torch.use_deterministic_algorithms(True)
        for seed in (17, 29, 43):
            torch.manual_seed(seed)
            baseline = ConverseUSRNet(num_iterations=2, num_blocks=1, backend='pytorch').cuda().train()
            candidate = copy.deepcopy(baseline)
            set_backend(candidate, 'cuda')
            x = torch.rand(1, 3, 8, 9, device='cuda')
            k = torch.softmax(torch.randn(1, 1, 49, device='cuda'), -1).reshape(1, 1, 7, 7)
            target = torch.rand(1, 3, 16, 18, device='cuda')
            optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (baseline, candidate)]
            for step in range(3):
                snapshots = []
                for model, optimizer in zip((baseline, candidate), optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    with frozen_model_reference():
                        out = model(x, k, 2)
                    loss = (out - target).square().mean()
                    loss.backward()
                    record = {'output': out.detach().clone(), 'loss': loss.detach().clone()}
                    record.update({f'gradient/{n}': p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None})
                    optimizer.step()
                    record.update({f'parameter/{n}': p.detach().clone() for n, p in model.named_parameters()})
                    for n, p in model.named_parameters():
                        for field, value in optimizer.state[p].items():
                            if torch.is_tensor(value):
                                record[f'optimizer/{n}/{field}'] = value.detach().clone()
                    snapshots.append(record)
                self.assertEqual(snapshots[0].keys(), snapshots[1].keys())
                failures = []
                for name, expected in snapshots[0].items():
                    row = dict(seed=seed, step=step, tensor=name,
                               **model_comparison(snapshots[1][name], expected, atol=3e-5, rtol=3e-5))
                    rows.append(row)
                    if not row['passed']:
                        failures.append(row)
                self.assertFalse(failures, failures)


if __name__ == '__main__':
    unittest.main(verbosity=2)
