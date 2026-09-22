#!/usr/bin/env python3
"""Four-metric DROID display, gated on complete source-bound evaluations."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/metrics_completion_v1/droid/recovery_v2'
OUT=ROOT/'paper/generated/metric_completion_droid'
METRICS=('standardized_mse','standardized_mae','raw_dinov2_l1','feature_cosine_distance')
LABELS=('Std. MSE','Std. MAE','Raw L1','Cosine dist.')
METHODS={'autoregressive':'Autoregressive','anchored_additive':'Additive anchor','bounded_additive':'Bounded additive','transport':'ShiftWM (ours)','unbounded_transport':r'No-$\tanh$ ablation','context_off':'No context','action_free':'No actions','persistence':'Persistence','official_one_step_shifted':'DINO-WM one-step (30 ep.)','matched_recursive_h10':'DINO-WM recursive (30 ep.)','official_raw_one_step':'DINO-WM raw one-step (100 ep.)','official_raw_recursive_h10':'DINO-WM raw recursive (100 ep.)'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,s):
    if not p.exists() or p.read_text()!=s:
        p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(s);t.replace(p)
def checked():
    marker=json.loads((REPORT/'completion.json').read_text())
    if marker.get('status')!='passed' or marker.get('completed_rows')!=34:raise ValueError('Incomplete study')
    if marker['finalization_sha256']!=sha(REPORT/'finalization.json') or marker['registration_sha256']!=sha(REPORT/'registration.json'):raise ValueError('Changed finalization')
    data=json.loads((REPORT/'finalization.json').read_text())
    if data['status']!='passed' or data['completed_learned_runs']!=33 or data['completed_persistence_runs']!=1:raise ValueError('Incomplete learned/persistence grid')
    if set(data['absolute_metrics'])!=set(METHODS) or data['population']!={'windows_per_run':1631,'episodes':141,'sessions':59}:raise ValueError('Wrong population')
    for rel,expected in data['source_dependencies'].items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=expected:raise ValueError('Changed source: '+rel)
    for row in data['absolute_metrics'].values():
        for metric in METRICS:
            v=np.asarray(row['mean_by_horizon'][metric])
            if v.shape!=(10,) or not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid scores')
    return data
def contrasts(data):
    result=[]
    for reference in ('autoregressive','unbounded_transport'):
        matches=[r for r in data['paired_comparisons'] if r['first_mode']=='transport' and r['second_mode']==reference]
        if len(matches)!=1:raise ValueError('Missing/duplicate comparator')
        for metric in METRICS:
            rows=[r for r in matches[0]['metrics'][metric] if r['horizon']==10]
            if len(rows)!=1:raise ValueError('Missing endpoint')
            r=rows[0];a=data['absolute_metrics']['transport']['mean_by_horizon'][metric][-1];b=data['absolute_metrics'][reference]['mean_by_horizon'][metric][-1]
            if not np.allclose([r['first_mean'],r['second_mean'],r['mean_difference']],[a,b,a-b],rtol=1e-10,atol=1e-12):raise ValueError('Paired mean mismatch')
            lo,hi=r['ci95']
            if not np.isfinite([lo,hi]).all() or lo>hi:raise ValueError('Invalid CI')
            result.append(dict(comparator=reference,metric=metric,difference=b-a,ci95=[-hi,-lo]))
    return result
def render(data):
    values=contrasts(data);absolute=data['absolute_metrics']
    table=r'''\begin{table}[!htb]\centering
\begingroup\fontsize{8.5}{10.2}\selectfont\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrrr@{}}\toprule
Method & MSE $\downarrow$ & MAE $\downarrow$ & Raw L1 $\downarrow$ & Cosine $\downarrow$ \\ \midrule
'''
    minima={k:min(r['mean_by_horizon'][k][-1] for r in absolute.values()) for k in METRICS};base=absolute['autoregressive']['mean_by_horizon']['standardized_mse'][-1]
    for i,(mode,label) in enumerate(METHODS.items()):
        if i==8:table+=r'\midrule\multicolumn{5}{@{}l}{Adapted official DINO-WM recipes; different training budgets} \\'+'\n'
        cells=[]
        for k in METRICS:
            v=absolute[mode]['mean_by_horizon'][k][-1];s=f'{v:.5f}';cells.append(r'\textbf{'+s+'}' if v==minima[k] else s)
        table+=' & '.join([label,*cells])+r' \\'+'\n'
    table+=r'\midrule'+'\n'
    for reference,label in [('autoregressive',r'ShiftWM gain vs. AR (\%)'),('unbounded_transport',r'ShiftWM gain vs. no-$\tanh$ (\%)')]:
        cells=[]
        for k in METRICS:
            b=absolute[reference]['mean_by_horizon'][k][-1];v=absolute['transport']['mean_by_horizon'][k][-1]
            gain=100*(b-v)/b;s=f'{gain:+.2f}'
            cells.append(r'\positivegain{'+s+'}' if gain>0 else s)
        table+=' & '.join([label,*cells])+r' \\'+'\n'
    table+=r'''\bottomrule\end{tabular}\endgroup
\caption{\textbf{DROID complementary errors at query step ten.} All 33 learned
checkpoints and persistence use the same 1,631 windows, 141 episodes and 59
development sessions. Average windows within episodes, then weight episodes and
three training seeds equally. MSE/MAE are training-standardized; L1/cosine use raw
DINOv2 features. Added metrics are secondary post-hoc analyses. Bold marks the
lowest point mean. The last two rows give relative error reductions; green
positive gains do not establish significance. External rows are
adapted recipes, not published leaderboard scores or matched-training-cost
architecture comparisons. All ten horizons and paired intervals are retained in
the source ledger. Recomputed MSE passes original per-window parity checks.}
\label{tab:droid-complementary-metrics}\end{table}
'''
    write(OUT/'scores.tex',table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','svg.hashsalt':'droid_complementary_metrics_v1','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#64748B'})
    fig,axes=plt.subplots(2,4,figsize=(5.5,2.18),dpi=220);fig.subplots_adjust(left=.15,right=.98,bottom=.24,top=.82,wspace=.40,hspace=1.1)
    for i,(k,label) in enumerate(zip(METRICS,LABELS)):
        axes[0,i].set_title(label,loc='left',fontsize=8.5,pad=9)
        for j,c in enumerate(('autoregressive','unbounded_transport')):
            ax=axes[j,i];ax.axvline(0,color='#64748B',lw=.75,zorder=1)
            r=next(r for r in values if r['metric']==k and r['comparator']==c);v=1000*r['difference'];lo,hi=np.array(r['ci95'])*1000;y=0
            ax.plot([lo,hi],[y,y],color='#243447',lw=1.2,zorder=2)
            for e in (lo,hi):ax.plot([e,e],[y-.08,y+.08],color='#243447',lw=.8)
            ax.scatter([v],[y],marker='o' if j==0 else 'D',s=22,color='#166534' if v>0 else '#B45339',zorder=3)
            ax.set_ylim(-.3,.3);ax.set_yticks([0],[('vs. AR' if j==0 else 'vs. no-tanh') if i==0 else '']);ax.tick_params(axis='y',length=0,pad=6);ax.tick_params(axis='x',labelsize=8,length=2,pad=3);ax.xaxis.set_major_locator(plt.MaxNLocator(3));ax.margins(x=.15)
    fig.text(.15,.055,'Comparator − ShiftWM (×1,000); each panel has its own scale',fontsize=8)
    for ext in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(OUT/('paired_metrics.'+ext),facecolor='white',metadata=metadata)
    plt.close(fig)
    from PIL import Image
    with Image.open(OUT/'paired_metrics.png') as im:im.convert('L').save(OUT/'paired_metrics_grayscale.png')
    evidence={'status':'completed','differences':values,'figure_inches':[5.5,2.18],'minimum_font_pt':8,'interval':'unadjusted95%session x seed paired bootstrap,10000draws','transform':'comparator minus ShiftWM; exact reversed CI endpoints; times1000','sources':{str(p.relative_to(ROOT)):sha(p) for p in (REPORT/'finalization.json',REPORT/'completion.json',REPORT/'registration.json',Path(__file__))}}
    write(OUT/'evidence.json',json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    write(OUT/'section.tex',r'''\subsubsection{Complementary DROID forecast errors}
\label{app:droid-complementary}
\input{generated/metric_completion_droid/scores.tex}
\begin{figure}[!htb]\centering
\includegraphics[width=\linewidth]{generated/metric_completion_droid/paired_metrics.pdf}
\caption{\textbf{Do gains extend beyond MSE?} Endpoint-error differences compare
ShiftWM with autoregression and the no-$\tanh$ ablation. Points show
comparator minus ShiftWM; intervals are unadjusted 95\% paired session--seed
bootstrap intervals. Each metric--comparator panel has its own horizontal scale, multiplied by
1,000 for readability. Positive means favor ShiftWM; intervals crossing zero
do not establish a reliable advantage. These secondary development analyses
do not change model selection or the primary endpoint.}
\label{fig:droid-complementary-metrics}\end{figure}
''')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (REPORT/'completion.json').exists():
        if (OUT/'evidence.json').exists():raise ValueError('Previously complete comparison lost its gate')
        write(OUT/'main_scope.tex','Spatial DROID currently reports native and pooled standardized MSE; its complementary-metric evaluation awaits the complete-study gate. IWS already reports standardized MSE, standardized MAE, raw DINOv2-feature L1 and cosine distance.\n')
        write(OUT/'section.tex',r'\paragraph{DROID complementary measurements.} The fixed-checkpoint four-metric evaluation awaits its complete-study gate. No partial score is inserted.'+'\n');print(json.dumps({'status':'pending','expected_rows':34}));return
    render(checked())
    write(OUT/'main_scope.tex',r'DROID and IWS report standardized MSE, standardized MAE, raw DINOv2-feature L1 and cosine distance. The DROID additions use all fixed checkpoints and the original development windows (Table~\ref{tab:droid-complementary-metrics}); its primary endpoint remains unchanged.'+'\n')
    print(json.dumps({'status':'completed','complete_rows':34,'methods':12,'metrics':4}))
if __name__=='__main__':main()
