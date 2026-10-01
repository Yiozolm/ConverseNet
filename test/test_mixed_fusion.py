"""CPU-only coverage/reference contracts; never a CUDA-fusion admission."""
from pathlib import Path
import copy
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.v4_mixed_fusion.gate import (finite_patterns, pad_specs, module_specs, valid_padding, record,
                                      admission_partition)


class MixedFusionContracts(unittest.TestCase):
    @staticmethod
    def admission_row(name, *, active, budget_pass=True):
        budget=dict(passed=budget_pass,baseline=dict(finite=True),candidate=dict(finite=True))
        return dict(kind='module',name=name,dtype='fp16',padding_mode='circular',scale=1,layout='contiguous',
                    context='no_grad',module_eps=1e-5 if active else 1e-8,passed=budget_pass,
                    expected_fused=active,fused_calls=2 if active else 0,
                    baseline_quantized_fp32_budget=copy.deepcopy(budget),candidate_quantized_fp32_budget=copy.deepcopy(budget),
                    baseline_budget_failure=not budget_pass,exact_current_module=dict(passed=True,candidate=dict(finite=True),baseline=dict(finite=True)),
                    repeated_exact=dict(passed=True),route_verified=True,fp32_shared_prior_and_mode_verified=True,
                    input_parameters_unchanged=True,parameter_identity_preserved=True,caller_state_preserved=True,
                    current_reference_finite=True,no_output_grad=True)

    def test_partition_keeps_failed_fallback_rows_failed(self):
        rows=[self.admission_row('active',active=True),self.admission_row('weak',active=False,budget_pass=False)]
        outcome=admission_partition(rows,expected_cases=2,expected_active=1)
        self.assertTrue(outcome['passed'])
        self.assertFalse(all(r['passed'] for r in rows))
        self.assertEqual(outcome['unchanged_fallback_budget_failures'],['weak'])

    def test_partition_cannot_ignore_active_budget_failure(self):
        self.assertFalse(admission_partition([self.admission_row('active',active=True,budget_pass=False)])['passed'])

    def test_partition_cannot_ignore_changed_fallback(self):
        active=self.admission_row('active',active=True)
        weak=self.admission_row('weak',active=False,budget_pass=False)
        weak['exact_current_module']['passed']=False
        self.assertFalse(admission_partition([active,weak])['passed'])

    def test_partition_rejects_candidate_only_failure(self):
        active=self.admission_row('active',active=True)
        fallback=self.admission_row('fallback',active=False)
        fallback['candidate_quantized_fp32_budget']['passed']=False
        fallback['passed']=False
        self.assertFalse(admission_partition([active,fallback])['passed'])

    def test_partition_rejects_nonfinite_even_when_records_match(self):
        active=self.admission_row('active',active=True)
        weak=self.admission_row('weak',active=False,budget_pass=False)
        weak['exact_current_module']['candidate']['finite']=False
        weak['exact_current_module']['baseline']['finite']=False
        self.assertFalse(admission_partition([active,weak])['passed'])
        contract=dict(kind='training_fallback',name='training',passed=True,baseline=[dict(finite=False)],candidate=[dict(finite=False)])
        self.assertFalse(admission_partition([active,contract])['passed'])

    def test_partition_cannot_mislabel_active_as_fallback(self):
        row=self.admission_row('active',active=True)
        row.update(expected_fused=False,fused_calls=0)
        self.assertFalse(admission_partition([row])['passed'])

    def test_partition_requires_complete_count_and_active_coverage(self):
        row=self.admission_row('active',active=True)
        self.assertFalse(admission_partition([row],expected_cases=2)['passed'])
        self.assertFalse(admission_partition([row],expected_active=2)['passed'])
        self.assertFalse(admission_partition([])['passed'])

    def test_all_finite_storage_patterns_include_both_signed_zeros(self):
        for dtype,count in ((torch.float16,63488),(torch.bfloat16,65280)):
            values=finite_patterns(dtype)
            self.assertEqual(values.numel(),count)
            self.assertTrue(torch.isfinite(values).all())
            zeros=values[values==0].float().view(torch.int32)
            self.assertEqual(set(zeros.tolist()),{0,-2147483648})

    def test_padding_boundaries(self):
        self.assertTrue(valid_padding(2,3,2,'circular'))
        self.assertFalse(valid_padding(2,3,2,'reflect'))
        self.assertTrue(valid_padding(1,1,6,'replicate'))
        self.assertTrue(valid_padding(1,1,6,'constant'))
        self.assertFalse(valid_padding(2,3,3,'circular'))
        self.assertFalse(valid_padding(5,7,0,'circular'))

    def test_matrix_unique_and_scope_visible(self):
        pads=list(pad_specs()); modules=list(module_specs())
        rows=pads+modules
        self.assertEqual(len({r['name'] for r in rows}),len(rows))
        self.assertEqual({r['dtype'] for r in pads},{'fp16','bf16'})
        self.assertEqual({r['mode'] for r in pads},{'circular','reflect','replicate','constant'})
        self.assertEqual(len(modules),384)
        real_broadcasts={(r['kb'],r['kc']) for r in modules if r['shape']==(2,3,5,7)}
        self.assertEqual(real_broadcasts,{(1,1),(1,3),(2,1),(2,3)})
        self.assertEqual({r['context'] for r in modules},{'no_grad','inference_mode','frozen'})
        self.assertTrue(any(r['shape']==(4,128,96,96) for r in modules))

    def test_padding_metadata_and_constant_zero_sign(self):
        x=torch.tensor([-0.,1.,-1.,0.],dtype=torch.float16).reshape(1,1,2,2)
        padded=F.pad(x.float(),(1,)*4,mode='constant',value=0)
        self.assertTrue(padded.is_contiguous())
        self.assertEqual(padded.storage_offset(),0)
        self.assertTrue((padded[...,0,:].view(torch.int32)==0).all())
        self.assertNotEqual(record(padded[...,1:-1,1:-1])['stride'],list(x.stride()))


if __name__=='__main__': unittest.main(verbosity=2)
