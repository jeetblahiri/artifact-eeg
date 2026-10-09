"""Figures tied to the complete causal follow-up output and declared comparators."""
import json
from common import ROOT,sha256,write_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/causal_revision';FIG=ROOT/'figures/causal_revision'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':230})

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def main():
    FIG.mkdir(parents=True,exist_ok=True);r=json.loads((OUT/'erp_assessment.json').read_text());old=json.loads((ROOT/'results/revision/summary.json').read_text());cfg=json.loads((ROOT/'causal_revision_config.json').read_text())
    lookup={(q['task'],q['method']):q for q in r['summary']};lookup.update({(q['task'],q['method']):q for q in old['summary']})
    methods=['parent','recipe_filter','recipe_brain_argmax','recipe_artifact80','ica_brain_argmax','ica_artifact80'];labels=['Parent','Filter only','Brain-only\nreference','Conservative\nreference','Brain-only ICA\non parent','Conservative ICA\non parent']
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
    for ax,task in zip(axes,['N170','P3']):
        ax.bar(range(6),[lookup[task,m]['mean_amplitude_uv'] for m in methods],color=['#32373c','#8a9ca6','#235f8c','#7296b0','#9b415a','#b994a0']);ax.axhline(0,c='#888',lw=.8)
        ax.set_xticks(range(6),labels,rotation=25,ha='right',fontsize=8);ax.set_ylabel('Measured contrast (µV)');ax.set_title(task+' | same ten evaluation people')
    fig.suptitle('Reference construction changes measured endpoints; parent is not cortical truth');fig.tight_layout();save(fig,'reference_stages')
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.5));control=json.loads((OUT/'target_contamination_control.json').read_text())
    for ax,task in zip(axes,['N170','P3']):
        targets=['parent','recipe_brain_argmax','recipe_artifact80'];original=[lookup[task,'target_'+m]['mean_amplitude_uv'] for m in targets]
        contaminated=[next(q['mean_amplitude_uv'] for q in control['summary'] if q['task']==task and q['target']==m) for m in targets]
        x=np.arange(3);ax.bar(x-.17,original,width=.34,label='Unchanged recording',color='#235f8c');ax.bar(x+.17,contaminated,width=.34,label='Shared synthetic EOG input',color='#9b415a')
        ax.set_xticks(x,['Parent','Brain-only recipe','Conservative recipe'],fontsize=9);ax.set_xlabel('Only training target changes');ax.set_ylabel('Fitted-output contrast (µV)');ax.set_title(task);ax.axhline(0,c='#888',lw=.8)
    axes[1].legend(fontsize=8);fig.suptitle('Controlled target intervention: identical learner, inputs and regularization');fig.tight_layout();save(fig,'matched_target_intervention')
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.8))
    targets=['parent','recipe_brain_argmax','recipe_artifact80'];names=['Parent','Brain-only','Conservative']
    for ax,task in zip(axes,['N170','P3']):
        d=json.loads((OUT/f'{task}_target_intervention.json').read_text());a=np.array([[d['losses'][t][v] for v in targets] for t in targets]);im=ax.imshow(a,vmin=0,vmax=.45,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]>.27 else 'black')
        ax.set_xticks(range(3),names,rotation=20,ha='right');ax.set_yticks(range(3),names);ax.set_xlabel('Scoring reference');ax.set_ylabel('Supervisory target');ax.set_title(task+' validation MSE')
    fig.subplots_adjust(bottom=.24,right=.84,wspace=.52);cax=fig.add_axes([.90,.24,.020,.64]);fig.colorbar(im,cax=cax,label='MSE after input-SD normalization');save(fig,'cross_target_loss_matrix')
    methods=['parent','icunet','simple_cnn','complex_cnn','femto'];old_methods=json.loads((ROOT/'results/revision/P3_metrics.json').read_text())['methods'];oldw=np.load(ROOT/'results/revision/P3_roi_contrasts.npy');neww=np.load(OUT/'P3_roi_contrasts.npy');idx=np.array(cfg['evaluation'])-1;time=np.arange(512)/256-.5
    fig,ax=plt.subplots(figsize=(6.8,3.8));colors=['#32373c','#235f8c','#ad6e22','#477f51','#9b415a'];names=['Parent','IC-U-Net','Benchmark simple CNN','Benchmark residual CNN','FemtoEOGClean']
    for m,color,label in zip(methods,colors,names):
        wave=oldw[idx,old_methods.index(m)].mean(0) if m in old_methods else neww[r['methods'].index(m),idx].mean(0)
        ax.plot(time*1000,wave,c=color,label=label,lw=1.6)
    ax.axvspan(300,600,color='#235f8c',alpha=.08);ax.set_xlim(-200,800);ax.set_xlabel('Time from event (ms)');ax.set_ylabel('P3 contrast (µV)');ax.legend(fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01));ax.set_title('P3 contrast on independent ERP CORE recordings',pad=59);fig.tight_layout();save(fig,'benchmark_model_p3')
    write_json(OUT/'figure_sources.json',dict(assessment_sha256=sha256(OUT/'erp_assessment.json'),matched_control_sha256=sha256(OUT/'target_contamination_control.json'),builder_sha256=sha256(__file__),plots=['reference_stages','matched_target_intervention','cross_target_loss_matrix','benchmark_model_p3']))

if __name__=='__main__':main()
