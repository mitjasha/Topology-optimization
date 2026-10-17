"""3D Hex8 FEM verification and dimensionality-specific neural model tests."""
from dataclasses import replace

import numpy as np
import pytest

from topopt.solver import optimize
from topopt.solver3d import Case3D, Model3D, hex8_stiffness, write_vtk
from topopt.mlnext.data3d import features3d, sample_cases


def test_hex8_symmetry_rigid_modes_and_strains():
    ke = hex8_stiffness(.5,.25,.4,.31)
    assert ke.shape == (24,24)
    assert np.allclose(ke,ke.T,atol=1e-12)
    for axis in range(3):
        rigid=np.zeros((8,3))
        rigid[:,axis]=1
        assert np.linalg.norm(ke@rigid.ravel())<1e-12
    eig=np.linalg.eigvalsh(ke)
    assert np.count_nonzero(eig<1e-9)==6
    assert eig[6]>0
    # Exact constant axial strain field u_x=epsilon*x (unit E).
    xyz=np.array([(0,0,0),(.5,0,0),(.5,.25,0),(0,.25,0),
                  (0,0,.4),(.5,0,.4),(.5,.25,.4),(0,.25,.4)])
    eps=.03
    disp=np.column_stack((eps*xyz[:,0],np.zeros(8),np.zeros(8))).ravel()
    energy=disp@ke@disp
    expected=(1-.31)/((1+.31)*(1-2*.31))*(.5*.25*.4)*eps**2
    assert np.isclose(energy,expected,rtol=1e-11)


def test_hex8_model_volume_compliance_equilibrium_and_gradient():
    c=Case3D(nx=4,ny=4,nz=4,volume=.45,radius=.59)
    m=Model3D(c)
    rho=np.full(m.ne,.45)
    a=m.analyze(rho)
    assert np.isclose(a["volume"],.45,atol=1e-10)
    assert a["compliance"]>0
    assert a["equilibrium_error"]<1e-8
    assert np.isclose(a["energy"],a["compliance"],rtol=1e-9)
    assert np.isclose(-sum(a["reaction"][2::3]),-1,atol=1e-9)
    rng=np.random.default_rng(123)
    d=rng.normal(size=m.ne)
    x=np.clip(.5+rng.normal(scale=.04,size=m.ne),.1,.9)
    h=1e-6
    finite_diff=(m.analyze(x+h*d)["compliance"]-m.analyze(x-h*d)["compliance"])/(2*h)
    analytical=m.analyze(x)["gradient"]@d
    assert np.isclose(finite_diff,analytical,rtol=2e-4,atol=1e-5)


def test_hex8_optimize_and_export(tmp_path):
    case=Case3D(nx=4,ny=4,nz=4,volume=.5,radius=.58)
    model=Model3D(case)
    result=optimize(model,max_iter=180,tolerance=.025,min_iter=15)
    assert result["converged"]
    assert result["equilibrium_error"]<1e-7
    assert abs(result["volume"]-case.volume)<1e-6
    initial=model.analyze(np.full(model.ne,case.volume))["compliance"]
    assert result["compliance"]<initial
    file=write_vtk(tmp_path/"density.vtk",case,result["rho"])
    raw=file.read_text()
    assert "DATASET STRUCTURED_POINTS" in raw
    assert "CELL_DATA 64" in raw
    assert len(raw.split("LOOKUP_TABLE default\n")[1].splitlines())==64


def test_3d_cases_and_features():
    samples=list(sample_cases(24,nx=8,ny=4,nz=4,seed=17))
    assert len(samples)==24
    assert {s for s,_ in samples}=={"train","validation","test"}
    c=samples[0][1]
    model=Model3D(c)
    x=np.full(model.ne,c.volume)
    feat=features3d(model,x)
    assert feat.shape==(11,4,4,8)
    assert np.all(feat[0]==c.volume)
    assert np.all(feat[4,:,:,:1]==1)
    assert np.all(feat[5,:,:,-1]==1)
    train_lengths=[c.length for s,c in samples if s=="train"]
    test_lengths=[c.length for s,c in samples if s=="test"]
    assert max(train_lengths)<min(test_lengths)


def test_volumetric_networks_forward_and_gradients():
    torch=pytest.importorskip("torch")
    from topopt.mlnext.models3d import build_3d_model
    torch.set_num_threads(1)
    inp=torch.rand(2,11,4,4,8)
    for name in ("unet3d","fno3d","gnn3d"):
        network=build_3d_model(name,11)
        prediction=network(inp)
        assert prediction.shape==(2,1,4,4,8)
        assert bool(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        assert any(p.grad is not None for p in network.parameters())


def test_2d_reference_modules_are_still_present():
    from topopt.solver import Case,Model
    c=Case(nx=8,ny=4)
    m=Model(c,backend="sparse")
    a=m.analyze(np.full(m.ne,c.volume))
    assert np.isfinite(a["compliance"]) and a["equilibrium_error"]<1e-7
