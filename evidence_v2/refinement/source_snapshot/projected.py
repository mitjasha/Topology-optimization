"""Smooth Heaviside density projection with an exact chain-rule gradient."""
from time import perf_counter
import numpy as np
from .solver import Model,optimize


class ProjectedModel(Model):
    def __init__(self,case,beta=1.0,eta=0.5,backend='sparse'):
        super().__init__(case,backend)
        if beta<=0 or not 0<eta<1:
            raise ValueError('Invalid projection parameters')
        self.beta,self.eta=beta,eta

    def projection(self,x):
        z=super().physical(x)
        b,e=self.beta,self.eta
        denominator=np.tanh(b*e)+np.tanh(b*(1-e))
        rho=(np.tanh(b*e)+np.tanh(b*(z-e)))/denominator
        derivative=b*(1-np.tanh(b*(z-e))**2)/denominator
        return rho,derivative

    def physical(self,x):
        return self.projection(x)[0]

    def analyze(self,x):
        x=np.asarray(x,dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1):
            raise ValueError('Invalid projected design')
        rho,derivative=self.projection(x)
        result=self.analyze_density(rho)
        result['gradient']=self.filter.T@(result['density_gradient']*derivative)
        # OC uses the derivative of the nonlinear physical-volume constraint.
        self.dv=self.filter.T@derivative/self.ne
        return result


def continuation(case,betas=(1,2,4,8,16),max_iter=200):
    start=perf_counter();x=None;stages=[];solves=0
    for beta in betas:
        model=ProjectedModel(case,beta=beta)
        result=optimize(model,initial=x,max_iter=max_iter,tolerance=0.01,min_iter=20)
        x=result['x'];solves+=result['fe_solves']
        stages.append({'beta':beta,'converged':result['converged'],'iterations':result['iterations'],
                       'compliance':result['compliance'],'volume':result['volume']})
    result.update(seconds=perf_counter()-start,fe_solves=solves,stages=stages)
    return model,result
