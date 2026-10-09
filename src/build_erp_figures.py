"""Export standalone manuscript figures from independently scored ERP results."""
import json
from common import ROOT,write_json,sha256
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['parent','hp0.5','hp1','hp2','regression','ica','femto','regression_guard','femto_guard']
LABELS=['Parent','HP 0.5 Hz','HP 1 Hz','HP 2 Hz','EOG regression','ICA/EOG','Frozen Femto','Regression + guard','Femto + guard']
COLORS={'parent':'#343a40','hp0.5':'#b07d28','hp1':'#b07d28','hp2':'#b07d28','regression':'#276797','ica':'#276797','femto':'#9b557d','regression_guard':'#276797','femto_guard':'#9b557d'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':230})

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def main():
    (ROOT/"figures").mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/erp_validation/summary.json';s=json.loads(source.read_text());cfg=s['config'];rows=s['summary'];lookup={(r['task'],r['method']):r for r in rows}
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),sharex='row');idx=np.arange(8)
    for col,task in enumerate(['N170','P3']):
        for line,(value,ci,label,mult) in enumerate([('amplitude_difference_uv','amplitude_difference_ci95','Amplitude change from parent (µV)',1),('frozen_ba_difference','frozen_ba_difference_ci95','Frozen-decoder change (percentage points)',100)]):
            ax=axes[line,col]
            for i,m in enumerate(METHODS[1:]):
                r=lookup[(task,m)];v=mult*r[value];lo,hi=np.array(r[ci])*mult
                ax.errorbar(v,i,xerr=np.array([[v-lo],[hi-v]]),fmt='o' if not m.endswith('_guard') else 's',color=COLORS[m],capsize=3,
                            mfc='white' if m.endswith('_guard') else COLORS[m])
            ax.axvline(0,c='#777777',ls='--',lw=1);ax.set_yticks(idx,LABELS[1:]);ax.invert_yaxis();ax.set_xlabel(label);ax.set_title(task+' · 10 evaluation participants')
    fig.suptitle('Observable preservation depends on the measured endpoint',fontweight='bold')
    fig.text(.5,.012,'Paired participant bootstrap 95% intervals; conditional on fixed development operations. Changes are not identified neural errors.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.955));save(fig,'erp_endpoint_changes')

    fig,axes=plt.subplots(1,2,figsize=(11.4,4.8));time=np.arange(512)/256-.5
    show=['parent','hp2','regression','femto','femto_guard'];styles=['-','--',':','-.','-']
    for ax,task in zip(axes,['N170','P3']):
        erp=np.load(ROOT/f'results/erp_validation/{task}_roi_contrasts.npy')[np.array(cfg['evaluation'])-1]
        for m,style in zip(show,styles):
            z=erp[:,METHODS.index(m)].mean(0);ax.plot(time*1000,z,ls=style,c=COLORS[m],lw=1.6,label=LABELS[METHODS.index(m)])
        win=cfg['tasks'][task]['window_s'];ax.axvspan(win[0]*1000,win[1]*1000,color='#777777',alpha=.12);ax.axhline(0,c='#aaaaaa',lw=.8)
        ax.set_xlim(-200,800);ax.set_xlabel('Time after stimulus (ms)');ax.set_ylabel('Event contrast (µV)');ax.set_title(task+' · '+('/'.join(cfg['tasks'][task]['roi'])))
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8.8,loc='lower center',bbox_to_anchor=(.5,.05),ncol=5)
    fig.suptitle('Independent event-linked measurements, with declared measurement windows',fontweight='bold')
    fig.text(.5,.012,'Face minus car and target minus nontarget; identical finite trials. Parent and guarded measurements can both contain peripheral voltage.',ha='center',fontsize=8.8)
    fig.tight_layout(rect=(0,.13,1,.93));save(fig,'erp_event_waveforms')

    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharex=True);probe_rows=[]
    for ax,task in zip(axes,['N170','P3']):
        p=json.loads((ROOT/f'results/erp_validation/{task}_probes.json').read_text())['rows'];probe_rows+=p
        for i,m in enumerate(METHODS):
            a=np.array([r['mean_endpoint_gain'] for r in p if r['method']==m]);ax.scatter(a,np.full(len(a),i),s=10,c=COLORS[m],alpha=.18)
            ax.plot(a.mean(),i,'s' if m.endswith('_guard') else 'o',c=COLORS[m],mfc='white' if m.endswith('_guard') else COLORS[m])
        ax.axvline(1,color='#777777',ls='--',lw=1);ax.set_yticks(np.arange(9),LABELS);ax.invert_yaxis();ax.set_xlabel('Known-voltage endpoint response gain');ax.set_title(task+' digital probe')
    fig.suptitle('Known digital voltage reveals directional attenuation',fontweight='bold')
    fig.text(.5,.012,'10 evaluation people × 6 signed amplitudes; points are participant/amplitude means. Digital controls do not establish cortical purity.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.945));save(fig,'erp_probe_response')

    cal=[r for r in s['conformal']['scores'] if r['split']=='calibration'];test=[r for r in s['conformal']['scores'] if r['split']=='evaluation'];q=s['conformal']['joint_q90']
    fig,ax=plt.subplots(figsize=(9.8,4.5))
    for offset,r,color,marker in [(0,cal,'#276797','o'),(12,test,'#9b557d','s')]:
        ax.scatter(np.arange(10)+offset,[v['joint_score'] for v in r],color=color,marker=marker,s=45,label='Calibration (n=10)' if offset==0 else 'Evaluation (n=10)')
    ax.axhline(q,c='#343a40',ls='--',lw=1.2,label=f'Joint 90% envelope q = {q:.2f}');ax.axhline(1,c='#b07d28',ls=':',lw=1,label='Illustrative tolerance = 1')
    ax.set_xticks(list(range(10))+list(range(12,22)),[str(r['subject']) for r in cal+test]);ax.set_xlabel('Participant identifier');ax.set_ylabel('Maximum standardized endpoint change');ax.legend(fontsize=9)
    ax.set_title('Selection-safe participant calibration of measured changes',fontweight='bold')
    fig.text(.5,.015,'Maximum across two tasks, amplitude/decoding and eight teachers. Marginal coverage assumes exchangeability; no neural-error coverage.',ha='center',fontsize=8.8)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,'erp_preservation_envelope')
    write_json(ROOT/'results/erp_validation/figure_sources.json',dict(summary_sha256=sha256(source),plots=['erp_endpoint_changes','erp_event_waveforms','erp_probe_response','erp_preservation_envelope'],
        evaluation_participants=cfg['evaluation'],intervals='Paired participant bootstrap 95%, descriptive, conditional on frozen development operations',
        figure_builder_sha256=sha256(__file__)))

if __name__=='__main__':main()
