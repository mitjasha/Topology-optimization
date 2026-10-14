from pathlib import Path
import tempfile
import unittest
import numpy as np
from topopt import Case,Model
from topopt.correlation import parse_dat,write_deck,evaluate


class CorrelationInterface(unittest.TestCase):
    def test_parser_requires_complete_independent_tables(self):
        text='displacements (vx,vy,vz)\n1 0 1D-3 0\n2 0 2E-3 0\nforces (fx,fy,fz)\n1 0 1 0\n2 0 -1 0\n'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'case.dat';path.write_text(text)
            parsed=parse_dat(path,2)
            self.assertEqual(parsed['u'][1,1],0.002)
            path.write_text(text.replace('2 0 -1 0\n',''))
            with self.assertRaises(ValueError):parse_dat(path,2)

    def test_export_preserves_modulus_thickness_and_force(self):
        m=Model(Case(nx=4,ny=2,load_y=0.37))
        design=np.linspace(0.1,0.9,m.ne)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'case.inp';write_deck(path,m,design,0.01)
            lines=path.read_text().splitlines()
            moduli=[float(lines[i+1].split(',')[0]) for i,line in enumerate(lines) if line=='*ELASTIC']
            expected=m.case.emin+(1-m.case.emin)*m.physical(design)**m.case.penal
            np.testing.assert_allclose(np.array(moduli)*0.01,expected,rtol=1e-12)
            a,b=lines.index('*CLOAD'),lines.index('*NODE PRINT,NSET=NALL')
            self.assertAlmostEqual(sum(float(line.split(',')[2]) for line in lines[a+1:b]),-1)

    def test_evaluation_rejects_changed_displacement(self):
        # This verifies the comparison logic only; it is not solver correlation.
        m=Model(Case(nx=4,ny=2));x=np.ones(m.ne);ref=m.analyze(x)
        u=np.zeros((m.nd//2,3));u[:,:2]=ref['displacement'].reshape(-1,2)
        rf=np.zeros(m.nd);rf[m.fixed]=ref['reaction']
        forces=np.zeros_like(u);forces[:,:2]=rf.reshape(-1,2)
        metrics,_=evaluate(m,x,{'u':u,'rf':forces})
        self.assertTrue(metrics['gate_pass'])
        metrics,_=evaluate(m,x,{'u':u*1.1,'rf':forces})
        self.assertFalse(metrics['gate_pass'])


if __name__=='__main__':unittest.main()
