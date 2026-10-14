"""Plane-stress Q4 FEM, linear density filter, SIMP and optimality criteria.

Independent implementation using isoparametric 2x2 Gauss integration.
Coordinates: x right, y up; arrays: (ny, nx), bottom row first.
All quantities are nondimensional; unit thickness and resultant force.
Dense assembly is intentional: this reference backend targets small examples.
"""
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class Case:
    nx: int = 16
    ny: int = 8
    length: float = 2.0
    height: float = 1.0
    nu: float = 0.3
    volume: float = 0.5
    load_y: float = 0.5
    load_span: float = 0.0
    radius: float = 0.19
    penal: float = 3.0
    emin: float = 1e-6

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2 or int(self.nx) != self.nx or int(self.ny) != self.ny:
            raise ValueError("Mesh counts must be integers >= 2")
        values = list(asdict(self).values())
        if not np.all(np.isfinite(values)):
            raise ValueError("All case parameters must be finite")
        if not (self.length > 0 and self.height > 0 and -1 < self.nu < 0.5):
            raise ValueError("Invalid dimensions or Poisson ratio")
        if not (0.001 < self.volume < 1 and 0 <= self.load_y <= 1):
            raise ValueError("Invalid volume or normalized load position")
        if not (0 <= self.load_span <= 1 and self.load_y-self.load_span/2 >= 0
                and self.load_y+self.load_span/2 <= 1):
            raise ValueError("Load patch must fit on the right edge")
        if not (self.radius > 0 and self.penal >= 1 and 0 < self.emin < 1):
            raise ValueError("Invalid SIMP/filter parameters")


def element_stiffness(hx, hy, nu):
    """Unit-E, unit-thickness Q4 stiffness, nodes BL/BR/TR/TL."""
    d = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]) / (1-nu**2)
    ke = np.zeros((8, 8))
    coords = np.array([[0, 0], [hx, 0], [hx, hy], [0, hy]])
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            dn = np.array([[-(1-eta), 1-eta, 1+eta, -(1+eta)],
                           [-(1-xi), -(1+xi), 1+xi, 1-xi]]) / 4
            jac = dn @ coords
            grad = np.linalg.solve(jac, dn)
            b = np.zeros((3, 8))
            b[0, 0::2], b[1, 1::2] = grad[0], grad[1]
            b[2, 0::2], b[2, 1::2] = grad[1], grad[0]
            ke += b.T @ d @ b * np.linalg.det(jac)
    return ke


class Model:
    def __init__(self, case):
        self.case = case
        nx, ny = case.nx, case.ny
        self.ne = nx * ny
        self.nd = 2 * (nx+1) * (ny+1)
        self.ke = element_stiffness(case.length/nx, case.height/ny, case.nu)
        nodes = []
        for j in range(ny):
            for i in range(nx):
                a = j*(nx+1)+i
                nodes.append([a, a+1, a+nx+2, a+nx+1])
        self.nodes = np.asarray(nodes)
        self.edof = np.stack((2*self.nodes, 2*self.nodes+1), axis=-1).reshape(self.ne, 8)
        self.rows = np.repeat(self.edof, 8, axis=1).ravel()
        self.cols = np.tile(self.edof, (1, 8)).ravel()
        fixed_nodes = np.arange(ny+1)*(nx+1)
        self.fixed = np.sort(np.r_[2*fixed_nodes, 2*fixed_nodes+1])
        self.free = np.setdiff1d(np.arange(self.nd), self.fixed)
        self.f = np.zeros(self.nd)
        # Interpolate load between adjacent right-edge nodes: continuous position,
        # exact total resultant and moment, no duplicate rounded load cases.
        y = case.load_y*ny
        lower = min(int(np.floor(y)), ny-1)
        weight = y-lower
        self.f[2*(lower*(nx+1)+nx)+1] = -(1-weight)
        self.f[2*((lower+1)*(nx+1)+nx)+1] = -weight
        if case.load_span > 0:
            # Consistent nodal loads for uniform traction over a fixed physical
            # patch. This avoids a shrinking point-load footprint in mesh studies.
            self.f[:] = 0
            a = (case.load_y-case.load_span/2)*case.height
            b = (case.load_y+case.load_span/2)*case.height
            dy = case.height/ny
            for j in range(ny):
                left, right = max(a,j*dy), min(b,(j+1)*dy)
                if right <= left:
                    continue
                upper = ((right-j*dy)**2-(left-j*dy)**2)/(2*dy)
                self.f[2*(j*(nx+1)+nx)+1] -= (right-left-upper)/(b-a)
                self.f[2*((j+1)*(nx+1)+nx)+1] -= upper/(b-a)
        xx, yy = np.meshgrid((np.arange(nx)+0.5)*case.length/nx,
                             (np.arange(ny)+0.5)*case.height/ny)
        xy = np.c_[xx.ravel(), yy.ravel()]
        dist = np.linalg.norm(xy[:, None, :]-xy[None, :, :], axis=2)
        h = np.maximum(0, case.radius-dist)
        self.filter = h / h.sum(axis=1, keepdims=True)
        self.dv = self.filter.T @ np.ones(self.ne) / self.ne

    def physical(self, x):
        return self.filter @ np.asarray(x).reshape(self.ne)

    def analyze(self, x):
        x = np.asarray(x, dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
            raise ValueError("Design variables must be finite and in [0, 1]")
        result = self.analyze_density(self.physical(x))
        result['gradient'] = self.filter.T @ result['density_gradient']
        return result

    def analyze_density(self, rho):
        """Analyze physical densities DIRECTLY, without applying a filter.

        Essential for binary reanalysis: filtering a 0/1 shape would silently
        replace it with another gray-density design. Void retains E_min.
        """
        rho = np.asarray(rho,dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(rho)) or np.any(rho < -1e-12) or np.any(rho > 1+1e-12):
            raise ValueError("Physical density must be finite and in [0,1]")
        # A normalized filter can sum to 1+roundoff on refined meshes.
        rho = np.clip(rho,0,1)
        c = self.case
        modulus = c.emin+(1-c.emin)*rho**c.penal
        k = np.zeros((self.nd, self.nd))
        np.add.at(k, (self.rows, self.cols), (modulus[:, None, None]*self.ke).ravel())
        u = np.zeros(self.nd)
        u[self.free] = np.linalg.solve(k[np.ix_(self.free, self.free)], self.f[self.free])
        residual = k@u-self.f
        ue = u[self.edof]
        energy = np.einsum('ei,ij,ej->e', ue, self.ke, ue)
        dc = -c.penal*(1-c.emin)*rho**(c.penal-1)*energy
        return dict(compliance=float(self.f@u), volume=float(rho.mean()),
                    density_gradient=dc, displacement=u, rho=rho,
                    equilibrium_error=float(np.linalg.norm(residual[self.free])),
                    reaction=residual[self.fixed], energy=float(np.dot(modulus, energy)))

    def project_volume(self, x):
        """Shift clipped design variables to satisfy physical mean density."""
        x = np.asarray(x, dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(x)):
            raise ValueError("Prediction contains non-finite values")
        lo, hi = -float(x.max())-1, 1-float(x.min())
        for _ in range(70):
            mid = (lo+hi)/2
            trial = np.clip(x+mid, 0.001, 1)
            if self.physical(trial).mean() > self.case.volume:
                hi = mid
            else:
                lo = mid
        return np.clip(x+(lo+hi)/2, 0.001, 1)


def optimize(model, initial=None, max_iter=160, tolerance=0.01, min_iter=15):
    """OC with physical-volume constraint; returns explicit stopping status.

    Convergence = max design-variable change below tolerance for 3 updates.
    This is an algorithmic stopping rule, not a global-optimality certificate.
    Final design is always reanalyzed and that solve is counted.
    """
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("Invalid stopping parameters")
    start = perf_counter()
    x = model.project_volume(np.full(model.ne, model.case.volume) if initial is None else initial)
    history, snapshots = [], []
    stable = 0
    converged = False
    for iteration in range(1, max_iter+1):
        result = model.analyze(x)
        lo, hi = 0.0, max(1.0, float(np.max(-result['gradient']/model.dv))*100)
        for _ in range(70):
            lam = (lo+hi)/2
            trial = np.clip(x*np.sqrt(np.maximum(0, -result['gradient'])/(model.dv*lam)),
                            np.maximum(0.001, x-0.2), np.minimum(1, x+0.2))
            if model.physical(trial).mean() > model.case.volume:
                lo = lam
            else:
                hi = lam
        change = float(np.max(np.abs(trial-x)))
        history.append(dict(iteration=iteration, compliance=result['compliance'],
                            volume=result['volume'], change=change))
        x = trial
        snapshots.append(x.copy())
        stable = stable+1 if change < tolerance else 0
        if iteration >= min_iter and stable >= 3:
            converged = True
            break
    final = model.analyze(x)
    return dict(x=x, rho=final['rho'], compliance=final['compliance'], volume=final['volume'],
                equilibrium_error=final['equilibrium_error'], converged=converged,
                iterations=iteration, fe_solves=iteration+1, history=history,
                snapshots=np.asarray(snapshots), seconds=perf_counter()-start)
