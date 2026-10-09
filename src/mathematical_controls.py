"""Numerical controls for additive mixing, source dependence and target deletion."""
import numpy as np
from scipy.stats import norm
from common import ROOT,config,write_json

def controls():
    rng=np.random.default_rng(config()['seed']);n=200000
    out={}
    # Additive observation, exact known-law nonlinear Bayes rule.
    s=rng.choice([-1.,1.],n);noise=rng.normal(0,1,n);x=s+noise
    bayes=np.tanh(x);linear=x/2
    out['additive_non_gaussian']=dict(linear_mse=float(np.mean((linear-s)**2)),
                                    exact_bayes_mse=float(np.mean((bayes-s)**2)))
    # Nonzero source correlation does not force a transfer penalty.
    pairs=[]
    for vs,va in [(1.,1.),(1.,4.)]:
        for rho in [-.8,-.4,0.,.4,.8]:
            cross=rho*np.sqrt(vs*va);vx=vs+va+2*cross
            wq=vs/(vs+va);wr=(vs+cross)/vx
            excess=(wq-wr)**2*vx
            pairs.append(dict(signal_variance=vs,artifact_variance=va,rho=rho,
                              synthetic_gain=wq,deployment_gain=wr,excess=excess))
    out['dependence_transfer']=pairs
    # Teacher projection deletes a task-bearing direction.
    latent=rng.choice([-1.,1.],n);neural=rng.normal(size=n)
    target=np.column_stack([neural,np.zeros(n)])
    full=np.column_stack([neural,.1*latent])
    stripped=target.copy()
    out['teacher_projection']=dict(per_sample_mse_stripped=float(np.mean((stripped-target)**2)),
                                   per_sample_mse_preserved=float(np.mean((full-target)**2)),
                                   task_accuracy_stripped=.5,task_accuracy_preserved=1.,
                                   construction='second coordinate carries balanced binary task; tie predicts fixed class')
    # Unknown scale produces nonlinear conditional means even with Gaussian source banks.
    var=np.asarray([.1,9.]);k=rng.integers(0,2,n);s=rng.normal(size=n);x=s+rng.normal(size=n)*np.sqrt(var[k])
    lik=norm.pdf(x[:,None],scale=np.sqrt(1+var))
    post=lik/lik.sum(1,keepdims=True)
    nonlin=x*np.sum(post/(1+var),1);lin=x/(1+var.mean());known=x/(1+var[k])
    out['unknown_scale']=dict(pooled_linear_mse=float(np.mean((lin-s)**2)),
                             exact_bayes_mse=float(np.mean((nonlin-s)**2)),
                             scale_informed_bayes_mse=float(np.mean((known-s)**2)))
    # Exact preservation and annihilation are incompatible on intersecting subspaces.
    u=np.array([1.,0.]);out['subspace_overlap']=dict(same_direction=u.tolist(),
        preservation_required='D u = u',suppression_required='D u = 0',
        any_output_sum_squared_errors_lower_bound=float(np.dot(u,u)/2))
    # Joint model remains linear in voltage; the statistical task need not be linear.
    write_json(ROOT/'results'/'mathematical_controls.json',out)
    assert out['additive_non_gaussian']['exact_bayes_mse'] < out['additive_non_gaussian']['linear_mse']
    assert all(abs(p['excess'])<1e-12 for p in pairs if p['signal_variance']==p['artifact_variance'])
    assert out['unknown_scale']['scale_informed_bayes_mse'] < out['unknown_scale']['exact_bayes_mse'] < out['unknown_scale']['pooled_linear_mse']
    return out

if __name__=='__main__':
    controls()
