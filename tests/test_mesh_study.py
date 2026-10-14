import unittest
import numpy as np
from topopt import Case,Model
from topopt.mesh_study import volume_binary


class MeshAndBinary(unittest.TestCase):
    def test_patch_load_resultant_and_centroid_across_meshes(self):
        for nx,ny in [(8,4),(16,8),(32,16)]:
            c=Case(nx=nx,ny=ny,load_y=0.37,load_span=0.25)
            m=Model(c);forces=m.f[1::2]
            yy=np.repeat(np.linspace(0,c.height,ny+1),nx+1)
            self.assertAlmostEqual(forces.sum(),-1,places=12)
            self.assertAlmostEqual(float(forces@yy),-0.37,places=12)

    def test_binary_reanalysis_does_not_filter(self):
        m=Model(Case(nx=6,ny=3,radius=0.5))
        rho=np.zeros(m.ne);rho[:m.ne//2]=1
        direct=m.analyze_density(rho)
        np.testing.assert_array_equal(direct['rho'],rho)
        self.assertFalse(np.allclose(m.analyze(rho)['rho'],rho))

    def test_volume_projection_is_binary_and_budgeted(self):
        b=volume_binary(np.array([0.1,0.8,0.5,0.5,0.9]),0.5)
        self.assertEqual(b.sum(),2)
        self.assertTrue(set(b)<=set([0,1]))
        self.assertEqual(b[1],1);self.assertEqual(b[4],1)
        self.assertEqual(volume_binary(np.ones(8),0.5).tolist(),[1,1,1,1,0,0,0,0])

    def test_subdivision_preserves_physical_density_volume(self):
        rho=np.array([[0,0.2],[0.5,1]])
        fine=np.repeat(np.repeat(rho,3,axis=0),3,axis=1)
        self.assertAlmostEqual(float(rho.mean()),float(fine.mean()))

    def test_invalid_patch_is_rejected(self):
        with self.assertRaises(ValueError):Case(load_y=0.1,load_span=0.5)

    def test_roundoff_is_tolerated_but_invalid_density_rejected(self):
        m=Model(Case(nx=4,ny=2))
        result=m.analyze_density(np.full(m.ne,1+2e-16))
        self.assertEqual(result['volume'],1)
        with self.assertRaises(ValueError):m.analyze_density(np.full(m.ne,1.001))


if __name__=='__main__':unittest.main()
