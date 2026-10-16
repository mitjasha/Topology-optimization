"""3D Hex8 FEM and SIMP optimization with the same OC interface as the 2D solver.

3D cells are indexed (nz, ny, nx), x fastest.  A left-end fixed face
supports a unit downward force on the opposite face.  Nondimensional units.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Case3D:
    nx: int = 8
    ny: int = 4
    nz: int = 4
    length: float = 2.0
    width: float = 1.0
    height: float = 1.0
    nu: float = .3
    volume: float = .5
    load_y: float = .5
    load_z: float = .5
    radius: float = .36
    penal: float = 3.0
    emin: float = 1e-6

    def __post_init__(self):
        if any(type(n) is not int or n < 2 for n in (self.nx, self.ny, self.nz)):
            raise ValueError("nx, ny, nz must be integers >= 2")
        scalars = (self.length, self.width, self.height, self.nu, self.volume,
                   self.load_y, self.load_z, self.radius, self.penal, self.emin)
        if not np.all(np.isfinite(scalars)):
            raise ValueError("All case parameters must be finite")
        if (min(self.length, self.width, self.height, self.radius) <= 0
                or not -1 < self.nu < .5 or not .001 < self.volume < 1
                or not 0 <= self.load_y <= 1 or not 0 <= self.load_z <= 1
                or self.penal < 1 or not 0 < self.emin < 1):
            raise ValueError("Invalid three-dimensional case parameters")


def hex8_stiffness(hx, hy, hz, nu):
    """Unit-E Hex8 stiffness using eight Gauss points, 3 displacement DOFs/node.

    Local nodes: (0,0,0), (1,0,0), (1,1,0), (0,1,0) and the same at z=1.
    Engineering strain ordering: xx, yy, zz, xy, yz, xz.
    """
    if min(hx, hy, hz) <= 0 or not -1 < nu < .5:
        raise ValueError("Invalid element dimensions or Poisson ratio")
    shear = 1 / (2 * (1 + nu))
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    constitutive = np.zeros((6, 6))
    constitutive[:3, :3] = lam
    constitutive[np.arange(3), np.arange(3)] += 2 * shear
    constitutive[3:, 3:] = np.eye(3) * shear
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    ke = np.zeros((24, 24))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            for zeta in (-1/np.sqrt(3), 1/np.sqrt(3)):
                corner = 1 + signs[:, 0] * xi
                side = 1 + signs[:, 1] * eta
                top = 1 + signs[:, 2] * zeta
                grad = np.column_stack((
                    signs[:,0] * side * top / (4*hx),
                    signs[:,1] * corner * top / (4*hy),
                    signs[:,2] * corner * side / (4*hz),
                ))
                b = np.zeros((6, 24))
                for a, (gx, gy, gz) in enumerate(grad):
                    p = 3*a
                    b[0,p] = gx
                    b[1,p+1] = gy
                    b[2,p+2] = gz
                    b[3,p:p+2] = (gy,gx)
                    b[4,p+1:p+3] = (gz,gy)
                    b[5,p:p+3:2] = (gz,gx)
                ke += b.T @ constitutive @ b * (hx*hy*hz/8)
    return (ke + ke.T) / 2


class Model3D:
    """Sparse Hex8 elasticity, physical cone filter, compliance and gradients."""
    def __init__(self, case):
        self.case = case
        nx, ny, nz = case.nx, case.ny, case.nz
        self.ne = nx*ny*nz
        self.nd = 3*(nx+1)*(ny+1)*(nz+1)
        self.ke = hex8_stiffness(case.length/nx, case.width/ny,
                                 case.height/nz, case.nu)

        def node(i,j,k):
            return (k*(ny+1)+j)*(nx+1)+i

        nodes = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    nodes.append([node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),
                                  node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),
                                  node(i+1,j+1,k+1),node(i,j+1,k+1)])
        self.nodes = np.asarray(nodes, dtype=np.int32)
        self.edof = (3*self.nodes[:,:,None] + np.arange(3)[None,None,:]).reshape(self.ne,24)
        self.rows = np.repeat(self.edof, 24, axis=1).ravel()
        self.cols = np.tile(self.edof, (1,24)).ravel()
        fixed_nodes = np.array([node(0,j,k) for k in range(nz+1) for j in range(ny+1)])
        self.fixed = (3*fixed_nodes[:,None] + np.arange(3)).ravel()
        self.free = np.setdiff1d(np.arange(self.nd), self.fixed)
        self.f = np.zeros(self.nd)
        sy, sz = case.load_y*ny, case.load_z*nz
        jy, kz = min(int(sy),ny-1), min(int(sz),nz-1)
        wy, wz = sy-jy, sz-kz
        for dy in (0,1):
            for dz in (0,1):
                force = (wy if dy else 1-wy)*(wz if dz else 1-wz)
                self.f[3*node(nx,jy+dy,kz+dz)+2] -= force

        zz, yy, xx = np.meshgrid(
            (np.arange(nz)+.5)*case.height/nz,
            (np.arange(ny)+.5)*case.width/ny,
            (np.arange(nx)+.5)*case.length/nx, indexing="ij")
        xyz = np.column_stack((xx.ravel(),yy.ravel(),zz.ravel()))
        neighbors = cKDTree(xyz).query_ball_point(xyz,case.radius)
        row,col,weights = [],[],[]
        for index, ids in enumerate(neighbors):
            ids = sorted(ids)
            row.extend([index]*len(ids))
            col.extend(ids)
            weights.extend(np.maximum(0,case.radius-np.linalg.norm(xyz[ids]-xyz[index],axis=1)))
        h = coo_matrix((weights,(row,col)),shape=(self.ne,self.ne)).tocsr()
        sums = np.asarray(h.sum(axis=1)).ravel()
        self.filter = diags(1/sums) @ h
        self.dv = np.asarray(self.filter.T @ np.ones(self.ne)).ravel()/self.ne
        self.backend = "sparse"

    def physical(self,x):
        return np.asarray(self.filter @ np.asarray(x,dtype=float).reshape(self.ne)).ravel()

    def assemble(self,rho):
        rho = np.asarray(rho,dtype=float).reshape(self.ne)
        c = self.case
        modulus = c.emin+(1-c.emin)*rho**c.penal
        values = (modulus[:,None,None]*self.ke).ravel()
        return coo_matrix((values,(self.rows,self.cols)),shape=(self.nd,self.nd)).tocsr()

    def analyze_density(self,rho):
        rho = np.asarray(rho,dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(rho)) or np.any(rho < -1e-12) or np.any(rho > 1+1e-12):
            raise ValueError("Physical density must be finite within [0,1]")
        rho = np.clip(rho,0,1)
        k = self.assemble(rho)
        u = np.zeros(self.nd)
        u[self.free] = spsolve(k[self.free][:,self.free].tocsc(),self.f[self.free])
        if not np.all(np.isfinite(u)):
            raise RuntimeError("Non-finite Hex8 FEM displacements")
        residual = k@u-self.f
        ue = u[self.edof]
        energy = np.einsum("ei,ij,ej->e",ue,self.ke,ue)
        c = self.case
        gradient = -c.penal*(1-c.emin)*rho**(c.penal-1)*energy
        return dict(compliance=float(self.f@u),volume=float(rho.mean()),
                    density_gradient=gradient,displacement=u,rho=rho,
                    equilibrium_error=float(np.linalg.norm(residual[self.free])),
                    reaction=residual[self.fixed],
                    energy=float(np.dot(c.emin+(1-c.emin)*rho**c.penal,energy)))

    def analyze(self,x):
        x = np.asarray(x,dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1):
            raise ValueError("Design must be finite and within [0,1]")
        result = self.analyze_density(self.physical(x))
        result["gradient"] = np.asarray(self.filter.T @ result["density_gradient"]).ravel()
        return result

    def project_volume(self,x):
        x = np.asarray(x,dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite volume prediction")
        lo,hi = -float(x.max())-1,1-float(x.min())
        for _ in range(70):
            mid = (lo+hi)/2
            trial = np.clip(x+mid,.001,1)
            if self.physical(trial).mean()>self.case.volume:
                hi=mid
            else:
                lo=mid
        return np.clip(x+(lo+hi)/2,.001,1)


def optimize3d(case, **kwargs):
    """Use the unchanged 2D/3D-independent optimality-criteria solver."""
    from topopt.solver import optimize
    return optimize(Model3D(case),**kwargs)


def write_vtk(path,case,density):
    """Write 3D cell-centered density in legacy VTK for ParaView."""
    rho = np.asarray(density,dtype=float).reshape(case.nz,case.ny,case.nx)
    if not np.all(np.isfinite(rho)):
        raise ValueError("Non-finite VTK output")
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    header = ("# vtk DataFile Version 3.0\n3D SIMP density\nASCII\n"
              "DATASET STRUCTURED_POINTS\n"
              f"DIMENSIONS {case.nx+1} {case.ny+1} {case.nz+1}\n"
              "ORIGIN 0 0 0\n"
              f"SPACING {case.length/case.nx:.12g} {case.width/case.ny:.12g} "
              f"{case.height/case.nz:.12g}\n"
              f"CELL_DATA {case.nx*case.ny*case.nz}\nSCALARS density float 1\n"
              "LOOKUP_TABLE default\n")
    path.write_text(header+"\n".join(f"{v:.9g}" for v in rho.ravel())+"\n",
                    encoding="ascii")
    return path
