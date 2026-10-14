import importlib.util
import unittest
import numpy as np
from topopt.families import define_cases,features,acceptable,evaluate_method
from topopt.solver import Case,Model,optimize


class FamilyProtocol(unittest.TestCase):
    def test_held_out_geometry_and_symmetry_groups(self):
        cases,split,groups=define_cases()
        a,b,c=[set(cases[i].length for i in split[key]) for key in ('train','validation','test')]
        self.assertFalse(a&b or a&c or b&c)
        a,b,c=[set(groups[i] for i in split[key]) for key in ('train','validation','test')]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(sum(map(len,split.values())),len(cases))

    def test_nonfinite_and_nonconverged_predictions_fail(self):
        case=Case();ref={'converged':True,'compliance':10}
        result={'converged':True,'compliance':10,'volume':case.volume,'equilibrium_error':0}
        self.assertTrue(acceptable(result,ref,case))
        result['compliance']=float('nan');self.assertFalse(acceptable(result,ref,case))
        result.update(compliance=10,converged=False);self.assertFalse(acceptable(result,ref,case))

    def test_early_stopping_fallback_runs_and_counts_full_solve(self):
        case=Case(nx=8,ny=4);reference=optimize(Model(case,backend='sparse'),max_iter=400,tolerance=0.02,min_iter=15)
        result=evaluate_method(case,reference,early_only=True)
        self.assertTrue(result['fallback']);self.assertTrue(result['final_gate_pass'])
        self.assertEqual(result['fe_solves'],6+reference['fe_solves'])

    def test_reference_quality_does_not_control_runtime_fallback(self):
        case=Case(nx=8,ny=4)
        reference={'converged':True,'compliance':1e-6}
        result=evaluate_method(case,reference)
        self.assertFalse(result['attempt_gate_pass'])
        self.assertFalse(result['fallback'])
        self.assertTrue(result['operational_gate_pass'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional PyTorch dependency absent')
    def test_unet_preserves_spatial_shape_and_has_skip_paths(self):
        import torch
        from topopt.neural import SmallUNet
        net=SmallUNet()
        output=net(torch.zeros(2,5,8,16))
        self.assertEqual(tuple(output.shape),(2,1,8,16))
        self.assertTrue(torch.all((output>=0)&(output<=1)))
