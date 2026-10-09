"""Standalone scientific figures for target-validity experiments."""
import json
from common import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                    'svg.fonttype':'none','figure.dpi':130,'savefig.dpi':240})

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight')
    plt.close(fig)

def main():
    (ROOT/"figures").mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'results/target_validity/summary.json').read_text())
    rows=source['known_source']['rows'];alpha=np.asarray([r['alpha'] for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.2))
    axes[0].plot(alpha,[r['benchmark_mse'] for r in rows],'o-',color='#4470a5',label='Surrogate MSE')
    axes[0].plot(alpha,[r['neural_mse'] for r in rows],'s-',color='#bc6656',label='True-neural MSE')
    axes[0].set_ylabel('MSE per temporal sample');axes[0].set_ylim(-.035,1.1);axes[0].legend(fontsize=9)
    axes[0].set_title('The optimal surrogate estimator')
    axes[1].plot(alpha,[r['benchmark_snr_db'] for r in rows],'o-',color='#4470a5',label='Benchmark SNR')
    axes[1].plot(alpha,[r['physiological_snr_db'] for r in rows],'s-',color='#2a927f',label='Physiological SNR')
    axes[1].set_ylabel('Power-SNR (dB)');axes[1].set_title('Identical observed waveform');axes[1].legend(fontsize=9)
    axes[2].plot(alpha,[100*r['population_optimal_task_accuracy'] for r in rows],'o-',color='#755398')
    axes[2].axhline(50,color='#8a929c',ls='--',lw=1);axes[2].set_ylim(45,92)
    axes[2].set_ylabel('Optimal task accuracy (%)');axes[2].set_title('Complete deletion loses the task')
    for ax in axes:
        ax.set_xlabel('Fraction of neural task component\nassigned to the artifact pool');ax.set_xticks(alpha)
    fig.suptitle('Perfect surrogate recovery can delete known neural task information',fontweight='bold')
    fig.text(.5,.01,'Stylized simulation with known sources; 100000 draws. Target allocations keep the observation exactly unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.93));save(fig,'surrogate_target_failure')

    teacher=json.loads((ROOT/'results/teacher_intervention/summary.json').read_text())['summary']
    variants=['parent','remove_8_30','retain_1_8','remove_1_4']
    labels=['Parent\n1–40 Hz','Remove\n8–30 Hz','Retain\n1–8 Hz','Remove\n1–4 Hz']
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.7))
    idx=np.arange(4)
    for offset,decoder,color in [(-.17,'CSP','#4470a5'),(.17,'bandpower','#2a927f')]:
        scores=[next(r['mean_accuracy']*100 for r in teacher if r['teacher']==v and r['decoder']==decoder) for v in variants]
        axes[0].bar(idx+offset,scores,width=.31,color=color,label=decoder)
        for i,score in enumerate(scores):
            axes[0].text(i+offset,score+.6,f'{score:.1f}',ha='center',fontsize=8)
    axes[0].axhline(25,color='#8a929c',ls='--',lw=1);axes[0].set_ylim(0,48)
    axes[0].set_ylabel('Participant mean task accuracy (%)');axes[0].legend(fontsize=9)
    axes[0].set_title('Teacher choices change task outcomes')
    energies=[next(r['deleted_scalp_energy_fraction']*100 for r in teacher if r['teacher']==v) for v in variants]
    axes[1].bar(idx,energies,width=.65,color='#bc6656');axes[1].set_ylim(0,66)
    for i,energy in enumerate(energies):
        axes[1].text(i,energy+1,f'{energy:.1f}',ha='center',fontsize=9)
    axes[1].set_ylabel('Deleted scalp-signal energy (%)');axes[1].set_title('Each teacher has zero self-reconstruction MSE')
    for ax in axes:
        ax.set_xticks(idx,labels)
    fig.suptitle('A reconstruction target does not validate its own task preservation',fontweight='bold')
    fig.text(.5,.015,'2592 real BCI trials, nine participant folds. Fixed FFT interventions; deleted scalp energy is not identified cortical energy.',ha='center',fontsize=8.7)
    fig.tight_layout(rect=(0,.065,1,.93));save(fig,'teacher_intervention')

    rank=[r for r in source['ranking_sensitivity'] if (r['f'],r['g']) in [('large_mlp','affine'),('tiny_cnn','affine')]]
    fig,ax=plt.subplots(figsize=(9.6,4.4))
    values=np.asarray([r['minimum_bias_fraction_target_rms']*100 for r in rank])
    y=np.arange(len(rank));colors=['#755398' if r['f']=='large_mlp' else '#cc8150' for r in rank]
    ax.barh(y,values,color=colors,height=.67)
    ax.set_yticks(y,[r['artifact']+': '+('MLP' if r['f']=='large_mlp' else 'Tiny CNN')+' vs affine' for r in rank])
    ax.invert_yaxis();ax.set_xlim(0,50)
    for i,v in enumerate(values):
        ax.text(v+.55,i,f'{v:.2f}%',va='center',fontsize=9)
    ax.set_xlabel('Minimum unconstrained bias RMS permitting a ranking tie (% of target RMS)')
    ax.set_title('Neural rankings require a justified target-bias bound',fontweight='bold')
    fig.text(.5,.015,'Sharp worst-direction sensitivity from saved predictions. These radii are hypothetical, not measured physiological impurity.',ha='center',fontsize=8.6)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,'target_ranking_sensitivity')

if __name__=='__main__':
    main()
