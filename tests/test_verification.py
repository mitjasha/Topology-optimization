import unittest
import numpy as np
from topopt.solver import Case, Model, element_stiffness, optimize
from topopt.learning import split_cases


class Verification(unittest.TestCase):
    def test_rigid_body_modes_and_symmetry(self):
        k = element_stiffness(0.2, 0.1, 0.3)
        np.testing.assert_allclose(k, k.T, atol=1e-14)
        for rigid in ([1,0,1,0,1,0,1,0], [0,1,0,1,0,1,0,1], [0,0,0,0.2,-0.1,0.2,-0.1,0]):
            np.testing.assert_allclose(k@rigid, 0, atol=1e-14)
        self.assertEqual(np.sum(np.linalg.eigvalsh(k) > 1e-10), 5)

    def test_constant_strain_energy(self):
        hx, hy, nu = 0.4, 0.2, 0.3
        u = np.array([0,0,hx,0,hx,0,0,0])*0.01
        energy = u@element_stiffness(hx,hy,nu)@u
        self.assertAlmostEqual(energy, hx*hy*0.01**2/(1-nu**2), places=13)

    def test_equilibrium_and_work(self):
        m = Model(Case(nx=8, ny=4, load_y=0.37))
        r = m.analyze(np.ones(m.ne))
        self.assertLess(r['equilibrium_error'], 1e-10)
        self.assertAlmostEqual(r['compliance'], r['energy'], places=9)
        self.assertAlmostEqual(r['reaction'][1::2].sum(), 1, places=10)
        self.assertAlmostEqual(r['reaction'][0::2].sum(), 0, places=10)

    def test_filtered_sensitivity_against_finite_difference(self):
        m = Model(Case(nx=6, ny=3, radius=0.5))
        x = np.random.default_rng(7).uniform(0.3,0.8,m.ne)
        g = m.analyze(x)['gradient']
        for i in (0, 7, 17):
            delta = np.zeros(m.ne); delta[i] = 1e-5
            fd = (m.analyze(x+delta)['compliance']-m.analyze(x-delta)['compliance'])/2e-5
            self.assertAlmostEqual(g[i]/fd, 1, places=5)

    def test_volume_and_objective_improvement(self):
        m = Model(Case(nx=8,ny=4,radius=0.3))
        start = m.analyze(np.full(m.ne,0.5))['compliance']
        result = optimize(m,max_iter=35)
        self.assertLess(result['compliance'],start)
        self.assertLess(abs(result['volume']-0.5),1e-8)
        self.assertEqual(result['fe_solves'], result['iterations']+1)

    def test_load_interpolation_and_filter(self):
        m = Model(Case(nx=8,ny=4,load_y=0.33))
        self.assertAlmostEqual(m.f.sum(),-1)
        np.testing.assert_allclose(m.physical(np.ones(m.ne)),1)
        np.testing.assert_allclose(m.physical(np.zeros(m.ne)),0)
        projected = m.project_volume(np.linspace(-1,2,m.ne))
        self.assertAlmostEqual(m.physical(projected).mean(),0.5,places=10)

    def test_case_split_has_no_overlap(self):
        split = split_cases(24)
        a,b,c = [set(v) for v in split.values()]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(a|b|c,set(range(24)))

    def test_mirror_groups_have_no_leakage(self):
        groups=np.minimum(np.arange(24),np.arange(24)[::-1])
        split=split_cases(24,groups=groups)
        a,b,c=[set(groups[ids]) for ids in split.values()]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(a|b|c,set(range(12)))

    def test_iteration_limit_is_not_called_convergence(self):
        result=optimize(Model(Case(nx=6,ny=3)),max_iter=1)
        self.assertFalse(result['converged'])

    def test_beam_refinement(self):
        # Slender solid cantilever: Euler-Bernoulli tip compliance = 4 L^3/H^3.
        # Q4 converges from a stiff coarse solution; clamp/shear effects remain.
        values=[]
        for nx,ny in ((8,2),(16,4),(32,8)):
            m=Model(Case(nx=nx,ny=ny,length=4,height=1,load_y=0.5))
            values.append(m.analyze(np.ones(m.ne))['compliance'])
        reference=4*4**3
        self.assertLess(abs(values[-1]-reference)/reference,0.06)
        self.assertLess(abs(values[-1]-reference),abs(values[0]-reference))


if __name__ == '__main__':
    unittest.main()
