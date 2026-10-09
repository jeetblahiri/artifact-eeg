"""Export source-backed revision figures (PNG/SVG), without manuscript files."""
import json
from common import ROOT,sha256,write_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'results/revision';FIG=ROOT/'figures/revision'
LABELS={'parent':'Parent','hp0.5':'HP 0.5 Hz','hp1':'HP 1 Hz','hp2':'HP 2 Hz','regression':'EOG regression',
 'ica':'ICA/EOG','mapping':'Mapping only','icunet':'IC-U-Net','femto':'FemtoEOGClean',
 'femto_guard':'Rectangular guard','femto_smooth':'Smooth guard','icunet_smooth':'IC-U-Net + smooth guard'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':230})

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    s=json.loads((OUT/'summary.json').read_text());a=json.loads((OUT/'reference_audit.json').read_text())
    fig,ax=plt.subplots(figsize=(6.4,2.8))
    keys=[('EOG','large_mlp','affine'),('EOG','tiny_cnn','affine'),('EMG','tiny_cnn','affine')]
    for i,key in enumerate(keys):
        r=next(r for r in a['family_intervals'] if (r['artifact'],r['f'],r['g'])==key)
        d=next(r for r in a['rankings'] if (r['artifact'],r['f'],r['g'])==key and r['profile']=='Released target')
        ax.plot([r['lower'],r['upper']],[i,i],c='#235f8c',lw=2)
        ax.plot(d['modified_risk_difference'],i,'o',c='#9b415a')
    ax.axvline(0,c='#888',ls='--',lw=1);ax.set_yticks(range(3),['EOG MLP − affine','EOG CNN − affine','EMG CNN − affine']);ax.invert_yaxis()
    ax.set_xlabel('MSE difference (dimensionless)');ax.set_title('Exact construction ranges; not confidence intervals')
    fig.tight_layout();save(fig,'reference_construction_rankings')
    fig,ax=plt.subplots(figsize=(6.4,3.6));methods=json.loads((OUT/'P3_metrics.json').read_text())['methods']
    wave=np.load(OUT/'P3_roi_contrasts.npy')[np.array(s['config']['evaluation'])-1].mean(0);t=np.arange(512)/256-.5
    for m,color,ls in [('parent','#32373c','-'),('icunet','#235f8c','--'),('femto','#9b415a','-'),('femto_smooth','#ad6e22',':')]:
        ax.plot(t*1000,wave[methods.index(m)],c=color,ls=ls,lw=1.7,label=LABELS[m])
    ax.axvspan(300,600,alpha=.08,color='#235f8c');ax.set_xlim(-200,800);ax.set_xlabel('Time from event (ms)');ax.set_ylabel('P3 contrast (µV)');ax.legend(fontsize=9,ncol=2)
    ax.set_title('Ten-person mean scalp contrast; not cortical ground truth');fig.tight_layout();save(fig,'independent_p3_waveforms')
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    for split,color,marker in [('calibration','#235f8c','o'),('evaluation','#9b415a','s')]:
        points=[(i+1,r['joint_score']) for i,r in enumerate(s['calibration']['scores']) if r['split']==split]
        ax.scatter(*np.array(points).T,c=color,marker=marker,label=split.capitalize())
    for key,ls in [('q90','--'),('q95','-.')]:
        q=s['calibration']['joint'][key]['quantile'];ax.axhline(q,c='#32373c',ls=ls,label=f'{key} = {q:.2f}')
    ax.axhline(1,c='#777',ls=':');ax.set_xlabel('Calibration participants 1–20; evaluation 21–30');ax.set_ylabel('Maximum scaled change');ax.legend(fontsize=9,ncol=2)
    ax.set_title('Family stress test; envelope width matters');fig.tight_layout();save(fig,'revised_change_envelope')
    p=json.loads((OUT/'smooth_probes.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10.2,4.2))
    show=['mapping','icunet','femto','femto_smooth']
    for ax,task in zip(axes,['N170','P3']):
        for i,m in enumerate(show):
            gains=np.array([r['gain'] for r in p['rows'] if r['task']==task and r['method']==m]);lo,hi=np.quantile(gains,[.05,.95])
            ax.plot([lo,hi],[i,i],c='#235f8c',lw=2);ax.plot(gains.mean(),i,'o',c='#9b415a')
        ax.axvline(1,c='#777',ls='--');ax.set_yticks(range(len(show)),[LABELS[m] for m in show]);ax.invert_yaxis();ax.set_xlabel('Finite paired voltage-response gain');ax.set_title(task+' broad direction')
    fig.suptitle('Probe dispersion: means and 5th–95th percentiles');fig.tight_layout();save(fig,'independent_probe_dispersion')
    write_json(OUT/'revision_figure_sources.json',dict(summary_sha256=sha256(OUT/'summary.json'),reference_audit_sha256=sha256(OUT/'reference_audit.json'),
        broad_probes_sha256=sha256(OUT/'smooth_probes.json'),builder_sha256=sha256(__file__),plots=['reference_construction_rankings','independent_p3_waveforms','revised_change_envelope','independent_probe_dispersion']))

if __name__=='__main__':main()
