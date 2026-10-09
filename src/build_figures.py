"""Publication-style vector and raster figures from saved study results."""
import csv
import json
from common import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','figure.dpi':130,'savefig.dpi':240})
COLORS=['#8a929c','#4470a5','#2a927f','#cc8150','#8358a4']

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight')
    plt.close(fig)

def main():
    (ROOT/"figures").mkdir(parents=True,exist_ok=True)
    methods=['identity','fourier','affine','tiny_cnn_mean_seeds','large_mlp_mean_seeds']
    labels=['Identity','Fourier linear','Full affine','Tiny CNN\n1905 params','Large MLP\n1.05M params']
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
    for ax,typ in zip(axes,['EOG','EMG','ECG']):
        rows={r['method']:r for r in csv.DictReader((ROOT/'results'/'mixtures'/typ/'summary.csv').open())}
        vals=[float(rows[m]['mse']) for m in methods]
        ax.bar(np.arange(5),vals,color=COLORS,width=.72)
        ax.set_xticks(np.arange(5),labels,rotation=30,ha='right')
        ax.set_title(typ+(' proxy' if typ=='ECG' else ' mixtures'))
        for i,v in enumerate(vals):
            ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9)
        ax.set_ylim(0,.73)
    axes[0].set_ylabel('Test MSE per temporal sample')
    fig.suptitle('Model complexity has source-dependent value',fontweight='bold')
    fig.text(.5,.005,'Source-disjoint mixtures; true power-SNR −7, −4, −1, +2 dB. Neural losses averaged across three fitted seeds.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95));save(fig,'synthetic_comparison')

    rows=list(csv.DictReader((ROOT/'results'/'real_task'/'subject_metrics.csv').open()))
    methods=['raw','minimal','regression','ica','femto_frozen','eog_only']
    fig,ax=plt.subplots(figsize=(10.5,5))
    s=json.loads((ROOT/'results'/'real_task'/'summary.json').read_text())['summary']
    for sub in range(1,10):
        vals=[float(next(r['accuracy'] for r in rows if int(r['subject'])==sub and r['method']==m and r['stratum']=='all')) for m in methods]
        ax.plot(np.arange(6),np.asarray(vals)*100,color='#9ba5ae',alpha=.5,lw=.8)
    means=[next(r['mean_accuracy'] for r in s if r['method']==m)*100 for m in methods]
    ax.plot(np.arange(6),means,'o-',color='#17445e',lw=2,label='Mean across participants')
    ax.axhline(25,color='#b55050',ls='--',lw=1,label='Four-class chance')
    ax.set_xticks(np.arange(6),['Acquisition raw','Minimal 1–40 Hz','EOG regression','ICA/EOG rejection','Published Femto','EOG only'])
    ax.set_ylabel('Held-out-participant accuracy (%)');ax.set_ylim(15,70)
    ax.set_title('Real-task effects differ across participants')
    ax.legend(frameon=False,loc='upper left');fig.tight_layout();save(fig,'real_task_participants')

    transfer=json.loads((ROOT/'results'/'benchmark_task_transfer'/'summary.json').read_text())['summary']
    eog={r['method']:r for r in csv.DictReader((ROOT/'results'/'mixtures'/'EOG'/'summary.csv').open())}
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    methods=['fourier','affine','tiny_cnn_mean_seeds','large_mlp_mean_seeds']
    names=['Fourier','Affine','Tiny CNN','Large MLP']
    for ax,decoder in zip(axes,['CSP','bandpower']):
        for i,(m,name) in enumerate(zip(methods,names)):
            y=next(r['mean_accuracy'] for r in transfer if r['decoder']==decoder and r['method']==m)*100
            x=float(eog[m]['mse'])
            offset=(5,5)
            if decoder=='bandpower' and m=='affine':
                offset=(-35,-22)
            if decoder=='bandpower' and m=='tiny_cnn_mean_seeds':
                offset=(7,14)
            ax.scatter(x,y,s=65,color=COLORS[i+1]);ax.annotate(name,(x,y),xytext=offset,textcoords='offset points',fontsize=9)
        ax.set_title(decoder+' decoder');ax.set_xlabel('EOG mixture MSE (lower is better)')
        ax.set_xlim(.042,.125);ax.set_ylim(32,46)
    axes[0].set_ylabel('Real motor-imagery accuracy (%)')
    fig.suptitle('A reconstruction ranking does not fix the task ranking',fontweight='bold')
    fig.text(.5,.005,'Post hoc transfer on the same BCI cohort; three neural seeds averaged. Source-person overlap cannot be excluded.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.045,1,.94));save(fig,'benchmark_task_ranking')

    controls=json.loads((ROOT/'results'/'mathematical_controls.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    axes[0].bar(['Affine','Exact Bayes'],[controls['additive_non_gaussian']['linear_mse'],controls['additive_non_gaussian']['exact_bayes_mse']],color=COLORS[1:3])
    axes[0].set_title('Additivity allows nonlinear gain');axes[0].set_ylabel('Known-law empirical MSE')
    for va,color in [(1,COLORS[1]),(4,COLORS[3])]:
        p=[r for r in controls['dependence_transfer'] if r['artifact_variance']==va]
        axes[1].plot([r['rho'] for r in p],[r['excess'] for r in p],'o-',label=f'Artifact variance {va}',color=color)
    axes[1].set_title('Dependence can have zero penalty');axes[1].set_xlabel('Source correlation');axes[1].set_ylabel('Deployment excess MSE');axes[1].legend(frameon=False,fontsize=8)
    axes[2].scatter([0,.005],[50,100],color=COLORS[1:3],s=70)
    axes[2].annotate('Teacher projection',(0,50),xytext=(5,8),textcoords='offset points',fontsize=8)
    axes[2].annotate('Preserved task direction',(.005,100),xytext=(-120,-15),textcoords='offset points',fontsize=8)
    axes[2].set_xlim(-.001,.007);axes[2].set_ylim(40,110)
    axes[2].set_title('Teacher loss can oppose task fidelity');axes[2].set_xlabel('Teacher-target MSE');axes[2].set_ylabel('Binary-task accuracy (%)')
    fig.tight_layout();save(fig,'mathematical_counterexamples')

if __name__=='__main__':
    main()
