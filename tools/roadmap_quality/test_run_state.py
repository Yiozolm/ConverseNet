"""CPU-only checks for stopping, safe serialization and resumable data/RNG state."""
from contextlib import ExitStack, redirect_stdout
import copy
import datetime
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_state


class RunStateTests(unittest.TestCase):
    def test_stability_needs_every_metric_and_the_declared_window(self):
        history = [dict(optimizer_steps=step,
                        rgb=dict(psnr_db=30., ssim=.9), y=dict(psnr_db=32., ssim=.92))
                   for step in range(0, 1001, 250)]
        self.assertTrue(run_state.stability(history, 1000)['satisfied'])
        self.assertFalse(run_state.stability(history[:-1], 750)['eligible'])
        shifted = copy.deepcopy(history)
        shifted[-1]['optimizer_steps'] = 1001
        self.assertFalse(run_state.stability(shifted, 1001)['eligible'])
        for space, metric, delta in (('rgb', 'psnr_db', .021), ('y', 'psnr_db', .021),
                                      ('rgb', 'ssim', .00051), ('y', 'ssim', .00051)):
            changed = copy.deepcopy(history)
            changed[-1][space][metric] += delta
            self.assertFalse(run_state.stability(changed, 1000)['satisfied'])
        history[-1]['y']['psnr_db'] = float('inf')
        self.assertFalse(run_state.stability(history, 1000)['satisfied'])

    def test_deadline_and_elapsed_budget(self):
        deadline = run_state.parse_deadline('2026-09-25T12:02:31Z')
        self.assertEqual(run_state.budget_reason(10., 5., None, monotonic_now=15.), 'max_wall_seconds')
        self.assertEqual(run_state.budget_reason(0., None, deadline, utc_now=deadline), 'deadline_utc')
        self.assertIsNone(run_state.budget_reason(10., 5., deadline, monotonic_now=14.,
                                                  utc_now=deadline-datetime.timedelta(seconds=1)))
        with self.assertRaises(ValueError):
            run_state.parse_deadline('2026-09-25T12:02:31')

    def test_rng_roundtrip_is_weights_only_serializable(self):
        previous = run_state.capture_rng(cuda=False)
        try:
            random.seed(91); np.random.seed(91); torch.manual_seed(91)
            state = run_state.capture_rng(cuda=False)
            expected = (random.random(), np.random.rand(3), torch.rand(3))
            stream = io.BytesIO()
            torch.save(state, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=True)
            run_state.restore_rng(restored, cuda=False)
            self.assertEqual(random.random(), expected[0])
            np.testing.assert_array_equal(np.random.rand(3), expected[1])
            self.assertTrue(torch.equal(torch.rand(3), expected[2]))
        finally:
            run_state.restore_rng(previous, cuda=False)

    def test_resume_rejects_data_recipe_source_and_counter_changes(self):
        config = dict(variant='before', seed=17, steps=10000, lr=1e-5, run_dir='old')
        report = dict(config=config, resume_recipe_sha256=run_state.recipe_hash(config, resume=True),
                      source_sha256={'code': 'a'}, split_sha256='split', input_checkpoint_sha256='initial',
                      dataset=dict(manifest_sha256='manifest', kernels_sha256={'kernel': 'hash'}))
        payload = dict(report, config_sha256=run_state.hash_json(config), resume_state_version=1,
                       optimizer_steps=0, next_data_step=0, state_dict={}, optimizer_state_dict={},
                       rng_state={}, metric_history=[], origin_initial_state_tensor_sha256='origin',
                       parent_run_dir='old', run_status='budget_stopped')
        self.assertEqual(run_state.validate_resume(payload, report), 0)
        for field in ('source_sha256', 'split_sha256', 'resume_recipe_sha256'):
            changed = copy.deepcopy(payload); changed[field] = 'changed'
            with self.assertRaises(ValueError):
                run_state.validate_resume(changed, report)
        changed = copy.deepcopy(payload); changed['dataset']['manifest_sha256'] = 'other'
        with self.assertRaises(ValueError):
            run_state.validate_resume(changed, report)
        changed = copy.deepcopy(payload); changed['next_data_step'] = 1
        with self.assertRaises(ValueError):
            run_state.validate_resume(changed, report)
        built_report = dict(report,backend={'build_manifest':{'inputs':{'compiler':'one'},'binary_sha256':'original'}})
        built_payload = dict(payload,backend_manifest=built_report['backend']['build_manifest'])
        self.assertEqual(run_state.validate_resume(built_payload,built_report),0)
        changed = copy.deepcopy(built_payload);changed['backend_manifest']['binary_sha256']='rebuilt'
        with self.assertRaises(ValueError):
            run_state.validate_resume(changed,built_report)
        extended = dict(config, steps=20000, run_dir='new', resume='latest.pth', max_wall_seconds=3600.)
        self.assertEqual(run_state.recipe_hash(extended, resume=True), report['resume_recipe_sha256'])
        self.assertNotEqual(run_state.recipe_hash(dict(extended, lr=2e-5), resume=True), report['resume_recipe_sha256'])

    def test_worker_budget_checkpoint_resume_matches_uninterrupted_cpu_updates(self):
        # Exercise the real worker/checkpoint control flow with a tiny CPU
        # model. CUDA execution/timing and image evaluation are explicit fakes;
        # this makes no claim about real model or GPU quality.
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        import models.converse_usrnet as model_module
        import evaluate_usrnet_quality as metrics
        import train_usrnet_dataset as worker
        import usrnet_training_data as data_module

        class Model(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(1))
            def cuda(self):
                return self

        seen = {}; events = {}; current = {'name': '', 'calls': 0}
        previous_rng = run_state.capture_rng(cuda=False)
        previous_flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                          torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
                          torch.backends.cuda.matmul.allow_tf32)
        previous_data_root = data_module.ROOT

        def train_step(model, optimizer, batch, args):
            optimizer.zero_grad(set_to_none=True)
            noise = random.random() + float(np.random.rand()) + torch.rand(1)
            loss = (model.weight * noise - batch[0].flatten()[0]).square().mean()
            loss.backward(); optimizer.step(); current['calls'] += 1
            return dict(loss=float(loss.detach()), grad_l2_norm=float(model.weight.grad.norm()),
                        loss_and_grad_finite=True, gradient_tensor_count=1, optimizer_applied=True,
                        microbatches=1, samples=1, training_step_wall_ms=1., training_step_cuda_span_ms=0.,
                        h2d_wall_ms=0., forward_backward_wall_ms=1., finite_check={'wall_ms':0.},
                        optimizer={'wall_ms':0.}, peak_memory={'allocated_bytes':0,'reserved_bytes':0})

        def evaluate(model, optimizer, protocol, ops, args, step, expected_ids, stop_reason=None):
            events[current['name']].append(('evaluation', step))
            if current['name'] == 'interrupt_eval' and step == 2:
                raise worker.BudgetStop('deadline_utc')
            value = float(model.weight.detach()[0])
            return dict(optimizer_steps=step, images=100, parameter_check={'finite':True},
                        validation_payload_sha256='validation', wall_s=.001, all_outputs_finite=True,
                        peak_memory={'allocated_bytes':0,'reserved_bytes':0}, per_image=[],
                        rgb=dict(psnr_db=30.+value,ssim=.9),y=dict(psnr_db=32.+value,ssim=.92))

        try:
            with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
                folder = Path(temporary).resolve()
                self.assertTrue(folder.is_relative_to(Path(tempfile.gettempdir()).resolve()))
                class Protocol:
                    def __init__(self, *args, **kwargs):
                        self.root = folder/'images'
                        self.train = [{'relative_path':f't{i}'} for i in range(900)]
                        self.validation = [{'relative_path':f'v{i}'} for i in range(100)]
                        self.metadata = dict(train_images=900,validation_images=100,
                                             manifest_sha256='manifest',kernels_sha256={'k':'hash'})
                    def train_batch(self, step, batch_size):
                        seen[current['name']].append(step)
                        events[current['name']].append(('data', step))
                        return (torch.full((1,3,8,8),float(step+1)), torch.ones(1,1,7,7), torch.ones(1,3,24,24))
                patches.enter_context(patch.object(model_module, 'ConverseUSRNet', Model))
                patches.enter_context(patch.object(metrics, 'checkpoint_layout', return_value={'strict':True}))
                patches.enter_context(patch.object(data_module, 'DatasetProtocol', Protocol))
                patches.enter_context(patch.object(worker, 'load_backend', return_value=(object(),
                    {'build_manifest':{'library':'cpu_test.pyd','binary_sha256':'binary'}})))
                patches.enter_context(patch.object(worker, 'source_hashes', return_value={}))
                original_file_hash = worker.file_hash
                patches.enter_context(patch.object(worker, 'file_hash', side_effect=lambda path:
                    'binary' if Path(path).name == 'cpu_test.pyd' else original_file_hash(path)))
                patches.enter_context(patch.object(worker, 'train_step', side_effect=train_step))
                patches.enter_context(patch.object(worker, 'evaluate', side_effect=evaluate))
                patches.enter_context(patch.object(run_state, 'budget_reason', side_effect=lambda started, limit, deadline:
                    'max_wall_seconds' if limit is not None and current['calls'] >= 2 else None))
                for name, replacement in {
                    'is_available':lambda:False, 'device_count':lambda:1,
                    'get_device_name':lambda *args:'CPU control-flow test',
                    'manual_seed_all':lambda *args:None, 'synchronize':lambda:None,
                    'reset_peak_memory_stats':lambda:None,
                    'get_rng_state_all':lambda:[torch.zeros(4,dtype=torch.uint8)],
                    'set_rng_state_all':lambda value:None,
                }.items():
                    patches.enter_context(patch.object(torch.cuda, name, replacement))

                def run(name, extra=()):
                    current.update(name=name,calls=0);seen[name]=[];events[name]=[]
                    directory = folder/name
                    command = ['worker','--root',str(root),'--init','scratch','--purpose','pilot',
                               '--variant','before','--steps','5','--eval-every','5',
                               '--batch-size','1','--microbatch-size','1','--patch-size','24',
                               '--run-dir',str(directory),*extra]
                    with patch.object(sys,'argv',command), redirect_stdout(io.StringIO()):
                        worker.main()
                    return json.loads((directory/'run.json').read_text()), torch.load(directory/'final.pth',weights_only=True)

                whole, whole_state = run('whole')
                stopped, stopped_state = run('stopped',('--max-wall-seconds','100'))
                self.assertEqual(stopped['status'],'budget_stopped')
                self.assertEqual(stopped_state['next_data_step'],2)
                self.assertTrue((folder/'stopped/latest.pth').is_file())
                resumed, resumed_state = run('resumed',('--resume',str(folder/'stopped/final.pth')))
                self.assertEqual(resumed['status'],'complete')
                self.assertEqual(seen['resumed'],[2,3,4])
                self.assertEqual(resumed['session_optimizer_steps'],3)
                self.assertEqual(resumed_state['next_data_step'],5)
                self.assertTrue(torch.equal(whole_state['state_dict']['weight'],resumed_state['state_dict']['weight']))
                self.assertTrue(torch.equal(whole_state['rng_state']['torch_cpu'],resumed_state['rng_state']['torch_cpu']))
                self.assertEqual(whole_state['metric_history'],resumed_state['metric_history'])
                _, whole_eval_state = run('whole_eval',('--eval-every','2'))
                interrupted, interrupted_state = run('interrupt_eval',('--eval-every','2'))
                self.assertEqual(interrupted['status'],'budget_stopped')
                self.assertEqual(interrupted_state['next_data_step'],2)
                self.assertEqual([v['optimizer_steps'] for v in interrupted_state['metric_history']],[0])
                _, resumed_eval_state = run('resumed_eval',('--eval-every','2','--resume',str(folder/'interrupt_eval/final.pth')))
                self.assertEqual(events['resumed_eval'][0],('evaluation',2))
                self.assertEqual(resumed_eval_state['metric_history'],whole_eval_state['metric_history'])
                self.assertTrue(torch.equal(whole_eval_state['state_dict']['weight'],resumed_eval_state['state_dict']['weight']))
        finally:
            run_state.restore_rng(previous_rng,cuda=False)
            data_module.ROOT = previous_data_root
            torch.use_deterministic_algorithms(previous_flags[0])
            torch.backends.cudnn.benchmark = previous_flags[1]
            torch.backends.cudnn.deterministic = previous_flags[2]
            torch.backends.cudnn.allow_tf32 = previous_flags[3]
            torch.backends.cuda.matmul.allow_tf32 = previous_flags[4]


if __name__ == '__main__':
    unittest.main(verbosity=2)
