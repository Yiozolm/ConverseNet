"""CPU checks for the real-image split, degradation convention and repeatability."""
import json
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from usrnet_training_data import DatasetProtocol, ROOT, degrade


METRICS = {}
MANIFEST = ROOT/'artifacts/dataset_training/split_900_100.json'


class USRNetTrainingData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = DatasetProtocol(MANIFEST, patch_size=96, scale=3, seed=20260917, noise_std=.01)

    def test_asymmetric_kernel_convolution_and_phase_zero(self):
        hr = np.random.default_rng(314159).random((18, 21, 3), dtype=np.float32)
        tensor = torch.from_numpy(hr.transpose(2, 0, 1).copy()).double()[None]
        records = []
        for index in (2, 4, 5):
            kernel = self.protocol.kernels[index-1]
            self.assertGreater(float(np.max(np.abs(kernel-kernel[::-1, ::-1]))), 1e-4)
            # conv2d computes correlation, so explicitly reverse both kernel
            # axes. Circular padding by three centers the repository 7x7 PSF.
            filt = torch.from_numpy(kernel.copy()).double().flip((-2, -1))[None, None]
            expected_hr = F.conv2d(F.pad(tensor, (3, 3, 3, 3), mode='circular'),
                                   filt.expand(3, 1, 7, 7), groups=3)
            for scale in (1, 3):
                actual = degrade(hr, kernel, scale, 0., np.random.default_rng(0))
                expected = expected_hr[0, :, ::scale, ::scale].permute(1, 2, 0).numpy()
                difference = np.abs(actual.astype(np.float64)-expected)
                np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
                records.append({'kernel':index,'scale':scale,'max_abs':float(difference.max()),
                                'relative_l2':float(np.linalg.norm(difference)/np.linalg.norm(expected))})
                if scale == 3:
                    wrong_phase = expected_hr[0, :, 1::3, 1::3].permute(1, 2, 0).numpy()
                    self.assertGreater(float(np.max(np.abs(actual-wrong_phase))), 1e-3)
        METRICS['degradation_vs_independent_float64_torch'] = records

    def test_train_batch_is_bitwise_reproducible_across_calls(self):
        for step in (0, 1, 449, 450):
            expected = self.protocol.train_batch(step, 2)
            self.protocol.train_batch(step+7, 2)
            actual = self.protocol.train_batch(step, 2)
            for a, e in zip(actual, expected):
                self.assertTrue(torch.equal(a, e), f'step {step} changed after an unrelated call')
            self.assertEqual(actual[0].shape, (2, 3, 32, 32))
            self.assertEqual(actual[1].shape, (2, 1, 7, 7))
            self.assertEqual(actual[2].shape, (2, 3, 96, 96))
        METRICS['training_reproducibility'] = {'steps':[0,1,449,450],'bitwise_equal':True}

    def test_validation_is_independent_of_training_seed(self):
        other = DatasetProtocol(MANIFEST, patch_size=96, scale=3, seed=17, noise_std=.01)
        first = list(self.protocol.validation_batches(7))
        second = list(other.validation_batches(7))
        self.assertEqual(len(first), len(second))
        count = 0
        for a, e in zip(first, second):
            self.assertEqual(a[0], e[0])
            count += len(a[0])
            for actual, expected in zip(a[1:], e[1:]):
                self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(count, 100)
        METRICS['validation_seed_independence'] = {'training_seeds':[20260917,17],
                                                  'images':count,'bitwise_equal':True}

    def test_split_has_no_encoded_or_pixel_hash_leakage(self):
        self.assertEqual((len(self.protocol.train),len(self.protocol.validation)),(900,100))
        for key in ('relative_path','file_sha256','rgb_sha256'):
            training = {row[key] for row in self.protocol.train}
            validation = {row[key] for row in self.protocol.validation}
            self.assertFalse(training & validation, key)
        self.assertGreaterEqual(min(min(r['width'],r['height']) for r in self.protocol.records),96)
        METRICS['split'] = {'train':900,'validation':100,'path_file_rgb_overlap':0,
                            'manifest_sha256':self.protocol.metadata['manifest_sha256']}


if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=2).result
    output = ROOT/'artifacts/dataset_training/data_protocol_checks.json'
    output.write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'metrics':METRICS,
                      'failures':[str(test)+'\n'+message for test,message in result.failures],
                      'errors':[str(test)+'\n'+message for test,message in result.errors]},indent=2)+'\n',encoding='utf-8')
    print('Saved',output)
    raise SystemExit(not result.wasSuccessful())
