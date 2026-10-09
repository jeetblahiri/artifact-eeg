"""Known-source and sharp target-uncertainty controls for surrogate validity."""
import csv
import json
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from common import ROOT,sha256,write_json

def settings():
    return json.loads((ROOT/'target_validity_config.json').read_text())

def risk_bounds(reference_mse,bias_radius,family_variance=0.):
    r=np.sqrt(max(0.,reference_mse-family_variance))
    return family_variance+max(0.,r-bias_radius)**2,family_variance+(r+bias_radius)**2

def ranking_radius(predictions_f,predictions_g,target):
    """Members are seed predictions; score mean losses, never an ensemble."""
    f=np.asarray(predictions_f,dtype=float);g=np.asarray(predictions_g,dtype=float)
    t=np.asarray(target,dtype=float)
    gap=float(np.mean((f-t)**2)-np.mean((g-t)**2))
    direction=f.mean(0)-g.mean(0)
    separation=float(np.sqrt(np.mean(direction**2)))
    return gap,separation,abs(gap)/(2*separation) if separation>0 else float('inf')

def known_source():
    c=settings();rng=np.random.default_rng(c['seed']);n=c['simulation_draws']
    u=rng.choice([-1.,1.],n)
    b=rng.normal(0,np.sqrt(c['simulation_background_variance']),n)
    low=rng.normal(0,np.sqrt(c['simulation_low_artifact_variance']),n)
    overlap=rng.normal(0,c['simulation_overlap_artifact_sd'],n)
    amp=c['simulation_task_amplitude'];sigma=c['simulation_overlap_artifact_sd']
    task=amp*u;z=task+overlap
    m=amp*np.tanh(amp*z/sigma**2)
    neural=np.column_stack([b,task,np.zeros(n)])
    artifact=np.column_stack([np.zeros(n),overlap,low])
    observed=neural+artifact
    # Basis modes have unit temporal RMS and are orthogonal, so sum of
    # coefficient squares equals per-temporal-sample waveform energy.
    time=np.arange(512)/256
    basis=np.sqrt(2)*np.asarray([np.sin(2*np.pi*f*time) for f in [20,12,2]])
    assert np.allclose(basis@basis.T/512,np.eye(3),atol=1e-12)
    dst=ROOT/'data'/'target_validity';dst.mkdir(parents=True,exist_ok=True)
    data_path=dst/'latent_coefficients.npz'
    np.savez_compressed(data_path,u=u,b=b,low=low,overlap=overlap,basis=basis)
    rows=[]
    # Symmetry permits conditioning on U=+1 for the even square. Independent
    # quadrature gives population values rather than calling Monte Carlo exact.
    conditional_mean_power=float(quad(lambda v:
        (amp*np.tanh(amp*(amp+sigma*v)/sigma**2))**2*np.exp(-v*v/2)/np.sqrt(2*np.pi),
        -12,12,epsabs=1e-11)[0])
    base_bayes_mse=amp**2-conditional_mean_power
    for alpha in c['teacher_deletion_fraction']:
        target=neural.copy();target[:,1]*=1-alpha
        extracted=artifact.copy();extracted[:,1]+=alpha*task
        assert np.allclose(target+extracted,observed)
        f=np.column_stack([b,(1-alpha)*m,np.zeros(n)])
        benchmark_mse=float(np.mean(np.sum((f-target)**2,1)))
        neural_mse=float(np.mean(np.sum((f-neural)**2,1)))
        pred=np.where(f[:,1]>0,1.,-1.)
        # At full deletion the output is independent of U. Exact optimal
        # accuracy is 1/2; an arbitrary fixed-class empirical score is saved.
        accuracy=float(np.mean(pred==u))
        rows.append(dict(alpha=alpha,benchmark_mse=benchmark_mse,neural_mse=neural_mse,
            theoretical_benchmark_mse_factor=(1-alpha)**2,
            theoretical_benchmark_mse=(1-alpha)**2*base_bayes_mse,
            theoretical_neural_mse=base_bayes_mse+alpha**2*conditional_mean_power,
            empirical_task_accuracy=accuracy,
            population_optimal_task_accuracy=.5 if alpha==1 else float(ndtr(amp/sigma)),
            teacher_removed_neural_energy=float(np.mean((alpha*task)**2)),
            benchmark_snr_db=float(10*np.log10(np.mean(np.sum(target**2,1))/np.mean(np.sum(extracted**2,1)))),
            physiological_snr_db=float(10*np.log10(np.mean(np.sum(neural**2,1))/np.mean(np.sum(artifact**2,1))))))
    return dict(rows=rows,draws=n,base_bayes_mse=base_bayes_mse,
        conditional_mean_power=conditional_mean_power,data_sha256=sha256(data_path),
        construction='Exact conditional means in an orthogonal three-mode simulation; known latent truth exists only in this control.',
        identity='Observed waveform is pointwise identical for every target allocation.')

def target_sensitivity():
    rows=[];bounds=[]
    c=settings()
    for typ in ['EOG','EMG','ECG']:
        t=np.load(ROOT/'data'/'mixtures'/f'{typ}_test.npz')['s'].astype(float)
        p=dict(np.load(ROOT/'results'/'mixtures'/typ/'predictions.npz'))
        cases={k:[p[k]] for k in ['identity','scalar','fourier','affine']}
        for family in ['tiny_cnn','large_mlp']:
            cases[family]=[p[f'{family}_seed{seed}'] for seed in [17,29,43]]
        rms=float(np.sqrt(np.mean(t*t)))
        for f,g in [('large_mlp','affine'),('tiny_cnn','affine'),('fourier','affine'),('tiny_cnn','large_mlp')]:
            fs,gs=np.asarray(cases[f],float),np.asarray(cases[g],float)
            gap,kappa,radius=ranking_radius(fs,gs,t)
            unit=(fs.mean(0)-gs.mean(0))/kappa
            delta=-np.sign(gap)*1.01*radius*unit
            # Delta=T-N. This hypothetical N is only an adversarial label.
            hypothetical=t-delta
            shifted=float(np.mean((fs-hypothetical)**2)-np.mean((gs-hypothetical)**2))
            assert np.allclose(shifted,gap+2*np.mean((fs.mean(0)-gs.mean(0))*delta),atol=1e-10)
            assert shifted*gap<0
            rows.append(dict(artifact=typ,f=f,g=g,reference_mse_difference=gap,
                family_mean_prediction_separation=kappa,minimum_bias_rms=radius,
                minimum_bias_fraction_target_rms=radius/rms,
                adversarial_check_bias_rms=1.01*radius,adversarial_check_difference=shifted,
                interpretation='Sharp unconstrained L2 target-bias sensitivity, not measured neural contamination.'))
        for method,values in cases.items():
            predictions=np.asarray(values,float)
            mse=float(np.mean((predictions-t)**2))
            variance=float(np.mean((predictions-predictions.mean(0))**2))
            for fraction in c['target_rms_uncertainty_fractions']:
                lo,hi=risk_bounds(mse,fraction*rms,variance)
                bounds.append(dict(artifact=typ,method=method,bias_fraction_target_rms=fraction,
                    reference_mse=mse,family_prediction_variance=variance,neural_mse_lower=lo,neural_mse_upper=hi))
    return rows,bounds

def main():
    dst=ROOT/'results'/'target_validity';dst.mkdir(parents=True,exist_ok=True)
    sim=known_source();rank,bounds=target_sensitivity()
    for name,rows in [('known_source',sim['rows']),('ranking_sensitivity',rank),('risk_bounds',bounds)]:
        with (dst/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    certificate=[dict(snr_power_db=db,identity_relative_rmse=10**(-db/20),
                      identity_relative_mse=10**(-db/10)) for db in [-7,-4,-1,2,5,10,15,20,25,30]]
    write_json(dst/'summary.json',dict(config=settings(),known_source=sim,ranking_sensitivity=rank,
        snr_identity_certificate=certificate,code_sha256=sha256(__file__),
        config_sha256=sha256(ROOT/'target_validity_config.json'),
        conclusion='Benchmark risk does not identify neural risk; all impurity radii are hypothetical.'))
    print('Completed known-source counterexample, sharp target-risk bounds and ranking-reversal controls.',flush=True)

if __name__=='__main__':
    main()
