"""FP32 training at scale 3 and large batch, checked against independent FP64.

Run directly to also write artifacts/training_scale3_batch_validation.json.
Pointwise budgets are fixed before measurement: output 3e-5/3e-5 and
gradient 5e-5/5e-5; weak outputs retain the existing stricter 1e-6/1e-5.
"""
import itertools
import json
import math
import sys
import unittest
from contextlib import contextmanager

import torch
from extension_loader import ROOT, load_extension
import test_fp32_training as common
from test_fp32_training import with_nearest_prior
from models.converse_core import converse2d_reference


RECORDS = []


def data(batch, channels, height, width, scale, kb=1, kc=1):
    x = torch.randn(batch, channels, height, width, device='cuda')
    prior = torch.randn(batch, channels, height*scale, width*scale, device='cuda')
    weight = torch.rand(kb, kc, 3, 3, device='cuda') / 9
    bias = torch.randn(1, channels, 1, 1, device='cuda')
    return tuple(t.requires_grad_() for t in (x, prior, weight, bias))


class TrainingScaleThreeBatch(unittest.TestCase):
    # Reuse the public comparison including every partial-gradient and
    # higher-order rule, without inheriting unrelated test methods.
    compare = common.FP32Training.compare

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(9214)

    @contextmanager
    def case(self, name, **settings):
        with self.subTest(case=name, **settings):
            record = dict(case=name, settings=settings, comparisons=[], status='running')
            RECORDS.append(record)
            self.record = record
            try:
                yield
            except Exception as exc:
                record.update(status='failed', error=str(exc))
                raise
            else:
                record['status'] = 'passed'

    def assert_numerical_close(self, actual, expected, *, atol, rtol, label):
        actual, expected = actual.detach().double(), expected.detach().double()
        difference = (actual-expected).abs()
        budget = atol + rtol*expected.abs()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        maximum = difference.max().item()
        relative = (difference.norm()/expected.norm().clamp_min(1e-30)).item()
        ratio = (difference/budget).max().item()
        numeric = lambda v: v if math.isfinite(v) else str(v)
        self.record['comparisons'].append(dict(
            tensor=label, shape=list(actual.shape), atol=atol, rtol=rtol, finite=finite,
            max_abs=numeric(maximum), relative_l2=numeric(relative),
            max_pointwise_budget_ratio=numeric(ratio),
            failed_elements=int((difference > budget).sum().item())))
        self.assertTrue(finite, label + ': nonfinite tensor')
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
            msg=lambda message: f'{label}: max_abs={maximum:.6g}, relative_l2={relative:.6g}, '
                                f'budget_ratio={ratio:.6g}\n{message}')

    def test_scale_three_broadcast_and_dynamic_kernels(self):
        for batch in (8, 32):
            for kb, kc in ((1, 1), (1, 3), (batch, 1), (batch, 3)):
                for nearest in (False, True):
                    with self.case('scale3', batch=batch, channels=3, height=64, width=80,
                                   scale=3, kb=kb, kc=kc, nearest=nearest, eps=1e-3):
                        x, prior, weight, bias = data(batch, 3, 64, 80, 3, kb, kc)
                        self.compare((x, weight, bias) if nearest else (x, prior, weight, bias),
                                     3, nearest=nearest)

    def test_scale_one_shared_input(self):
        for batch in (8, 32):
            for kb, kc in ((1, 1), (batch, 3)):
                with self.case('shared_x_prior', batch=batch, channels=3, height=64,
                               width=80, scale=1, kb=kb, kc=kc, eps=1e-3):
                    x, _, weight, bias = data(batch, 3, 64, 80, 1, kb, kc)
                    self.compare((x, weight, bias), 1, nearest=True)

    def test_scale_three_partial_and_higher_order_gradients(self):
        for needs in itertools.product((False, True), repeat=4):
            if any(needs):
                with self.case('scale3_higher_order', needs=needs, batch=2, channels=3,
                               height=3, width=4, scale=3, eps=1e-3):
                    args = tuple(t.detach().requires_grad_(need)
                                 for t, need in zip(data(2, 3, 3, 4, 3, 1, 3), needs))
                    self.compare(args, 3, higher=True)

    def compare_shared_kernel_to_individual_references(self, args, scale, nearest=False):
        """Keep real CUDA batch size; release its graph before FP64 sample VJPs.

        This bounds reference memory for B32/C32/s3 and explicitly checks that
        shared kernel and bias gradients equal the sum over individual samples.
        """
        op = torch.ops.converse2d.forward
        output = with_nearest_prior(op, *args, scale, 1e-3) if nearest else op(*args, scale, 1e-3)
        upstream = torch.randn_like(output)/output.numel()**.5
        gradients = torch.autograd.grad(output, args, upstream)
        output = output.detach()
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(all(g.dtype == a.dtype for g, a in zip(gradients, args)))
        image_count = len(args)-2
        expected_shared = [torch.zeros_like(a, dtype=torch.float64) for a in args[-2:]]
        self.assertEqual(args[-2].shape[0], 1)
        for sample in range(args[0].shape[0]):
            refs = tuple((a[sample:sample+1] if i < image_count else a)
                         .detach().double().requires_grad_() for i, a in enumerate(args))
            expected = (with_nearest_prior(converse2d_reference, *refs, scale, 1e-3) if nearest
                        else converse2d_reference(*refs, scale, 1e-3))
            expected_grads = torch.autograd.grad(expected, refs, upstream[sample:sample+1].double())
            self.assert_numerical_close(output[sample:sample+1], expected,
                                        atol=3e-5, rtol=3e-5, label=f'output/sample{sample}')
            for index in range(image_count):
                self.assert_numerical_close(gradients[index][sample:sample+1], expected_grads[index],
                                            atol=5e-5, rtol=5e-5,
                                            label=f'gradient{index}/sample{sample}')
            for accumulator, gradient in zip(expected_shared, expected_grads[-2:]):
                accumulator.add_(gradient)
        for name, actual, expected in zip(('weight', 'bias'), gradients[-2:], expected_shared):
            self.assert_numerical_close(actual, expected, atol=5e-5, rtol=5e-5,
                                        label=f'{name}/sum_of_sample_gradients')

    def test_full_channels_and_batch_gradient_accumulation(self):
        for batch, channels, height, width, nearest in (
                (8, 3, 64, 80, True), (32, 32, 64, 80, False), (4, 32, 256, 256, True)):
            with self.case('sample_gradient_sum', batch=batch, channels=channels, height=height,
                           width=width, scale=3, kb=1, kc=channels, nearest=nearest, eps=1e-3):
                x, prior, weight, bias = data(batch, channels, height, width, 3, 1, channels)
                args = (x, weight, bias) if nearest else (x, prior, weight, bias)
                self.compare_shared_kernel_to_individual_references(args, 3, nearest)

    def test_weak_regularization_large_batches(self):
        for batch, amplitude in itertools.product((8, 32), (0., 1e-6, 1e-3)):
            with self.case('weak_regularization', batch=batch, channels=3, height=5, width=7,
                           scale=3, eps=1e-8, bias=-40., amplitude=amplitude, kb=1, kc=1):
                args = data(batch, 3, 5, 7, 3)
                with torch.no_grad():
                    args[0].mul_(1e-5)
                    args[1].mul_(1e-5)
                    args[2].mul_(amplitude)
                    args[3].fill_(-40.)
                refs = tuple(t.detach().double().requires_grad_() for t in args)
                actual = torch.ops.converse2d.forward(*args, 3, 1e-8)
                expected = converse2d_reference(*refs, 3, 1e-8)
                upstream = torch.randn_like(actual)*1e-5
                actual_grads = torch.autograd.grad(actual, args, upstream)
                expected_grads = torch.autograd.grad(expected, refs, upstream.double())
                self.assert_numerical_close(actual, expected, atol=1e-6, rtol=1e-5, label='weak output')
                for index, (a, e) in enumerate(zip(actual_grads, expected_grads)):
                    self.assert_numerical_close(a, e, atol=5e-5, rtol=5e-5, label=f'weak gradient{index}')


if __name__ == '__main__':
    # Explicit suite avoids rerunning the imported base test class.
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainingScaleThreeBatch)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(seed=9214, torch=torch.__version__, tests=result.testsRun,
                  passed=result.wasSuccessful(), cases=RECORDS,
                  failures=[dict(case=str(t), error=e) for t, e in result.failures],
                  errors=[dict(case=str(t), error=e) for t, e in result.errors])
    target = ROOT/'artifacts/training_scale3_batch_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved', target, flush=True)
    sys.exit(not result.wasSuccessful())
