import unittest
import numpy as np
from topopt.solver import Case,Model,optimize
from topopt.projected import ProjectedModel


class SparseAndProjection(unittest.TestCase):
    def test_sparse_matches_dense_fields_gradients_and_optimization(self):
        case=Case(nx=8,ny=4,load_y=0.37,load_span=0.2)
        dense,sparse=Model(case),Model(case,backend='sparse')
        x=np.random.default_rng(2).uniform(0.3,0.8,dense.ne)
        a,b=dense.analyze(x),sparse.analyze(x)
        for key in ('rho','compliance','displacement','reaction','gradient'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-9,atol=1e-9)
        a,b=optimize(dense,max_iter=25),optimize(sparse,max_iter=25)
        np.testing.assert_allclose(a['x'],b['x'],rtol=1e-8,atol=1e-8)

    def test_sparse_filter_storage_scales_with_local_neighbours(self):
        model=Model(Case(nx=64,ny=32),backend='sparse')
        self.assertLess(model.filter.nnz,model.ne**2//10)
        np.testing.assert_allclose(model.physical(np.ones(model.ne)),1,atol=1e-14)

    def test_projected_compliance_and_volume_gradients(self):
        model=ProjectedModel(Case(nx=6,ny=3,radius=0.4),beta=4)
        x=np.random.default_rng(3).uniform(0.25,0.75,model.ne)
        result=model.analyze(x);dv=model.dv.copy()
        for index in (0,7,17):
            delta=np.zeros(model.ne);delta[index]=1e-6
            a,b=model.analyze(x+delta),model.analyze(x-delta)
            self.assertAlmostEqual(result['gradient'][index]/((a['compliance']-b['compliance'])/2e-6),1,places=5)
            self.assertAlmostEqual(dv[index],(a['volume']-b['volume'])/2e-6,places=7)

    def test_projected_physical_volume_constraint(self):
        for beta in (1,4,16):
            model=ProjectedModel(Case(nx=8,ny=4,volume=0.35),beta=beta)
            x=model.project_volume(np.linspace(-1,2,model.ne))
            self.assertAlmostEqual(model.physical(x).mean(),0.35,places=9)
