"""Complete-only reporting of the fixed development candidate experiment."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/iclr_review_2026-09-22/action_ranking'
OUT=ROOT/'paper/generated/iclr_review_v1'
MODES=['autoregressive','bounded_additive','transport','unbounded_transport']
LABELS=['AR','Additive','Bounded','Unbounded'];COLORS=['#59616D','#AE7922','#255E91','#82517E']
METRICS=['forecast_mse_mean','rho_physical','rho_actual_feature','normalized_physical_regret']

def main():
    reg=json.loads((REPORT/'registration.json').read_text());digest=hashlib.sha256((REPORT/'registration.json').read_bytes()).hexdigest()
    summary={};sources={}
    for task in ['pusht','reacher']:
        path=REPORT/(task+'.json');d=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert d['status']=='complete'and len(d['records'])==8 and d['registration_sha256']==digest
        assert [r['episode_id']for r in d['records']]==[r['episode_id']for r in reg['cases'][task]]
        summary[task]={}
        for mode in MODES:
            summary[task][mode]={}
            for metric in METRICS:
                x=np.array([[r['models'][f'{mode}_s{s}'][metric]for r in d['records']]for s in range(3)],float)
                assert x.shape==(3,8)and np.isfinite(x).all()
                rng=np.random.default_rng(173);seed=rng.integers(3,size=(10000,3));case=rng.integers(8,size=(10000,8))
                draws=x[seed[:,:,None],case[:,None,:]].mean((1,2))
                summary[task][mode][metric]={'mean':float(x.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'seed_means':x.mean(1).tolist()}
    plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(5.5,3.5),layout='constrained')
    for col,task in enumerate(['pusht','reacher']):
        for row,metric in enumerate(['forecast_mse_mean','rho_physical']):
            ax=axes[row,col]
            for i,mode in enumerate(MODES):
                v=summary[task][mode][metric];m=v['mean'];lo,hi=v['ci95']
                ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',color=COLORS[i],capsize=3,ms=4)
            ax.set_xticks(range(4),LABELS,rotation=25,ha='right');ax.set_xlim(-.45,3.45)
            if row==0:ax.set_title('PushT'if task=='pusht'else'Reacher',loc='left',weight='bold')
            if col==0:ax.set_ylabel('Forecast MSE ↓'if row==0 else'Physical ranking ρ ↑')
            ax.set_ylim((0,.8)if row==0 else(-.05,1.05));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',lw=.5)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'action_ranking.{ext}',dpi=240)
    plt.close(fig)
    table=[r'\begin{table}[tbp]\centering\small',r'\begin{tabular}{@{}llrrrr@{}}\toprule',r'Task & Method & MSE $\downarrow$ & $\rho_{\rm phys}$ $\uparrow$ & $\rho_{\rm feat}$ $\uparrow$ & Regret $\downarrow$ \\ \midrule']
    for task in summary:
        for mode,label in zip(MODES,LABELS):table.append(('PushT'if task=='pusht'else'Reacher')+' & '+label+' & '+' & '.join(f"{summary[task][mode][m]['mean']:.3f}"for m in METRICS)+r' \\')
    table.extend([r'\bottomrule\end{tabular}',r'\caption{Development-only candidate-ranking diagnostic. Each task uses eight metadata-selected episodes, 33 shared native-action candidates (32 uniform random sequences and one zero sequence), 25 executed future calls and all three selected training seeds. MSE compares predicted and actual terminal features. Spearman correlations compare predicted goal cost with physical goal distance ($\rho_{\rm phys}$) and actual feature goal cost ($\rho_{\rm feat}$). Regret is the selected candidate\textquotesingle s physical distance minus the best candidate distance, divided by the candidate distance range. Means weight cases and seeds equally. Random-candidate ranking is not CEM performance, and the diagnostic does not select new models.}',r'\label{tab:development-action-ranking}\end{table}'])
    (OUT/'action_ranking_table.tex').write_text('\n'.join(table)+'\n')
    (OUT/'action_ranking_figure.tex').write_text(r'''\begin{figure}[tbp]\centering
\includegraphics[width=\linewidth]{generated/iclr_review_v1/action_ranking.pdf}
\caption{Prediction error and physical action ranking answer different questions.
All four selected model families score the same executed development candidates.
Points average eight starting cases and three training seeds; bars are descriptive
95\% crossed seed--case bootstrap intervals (10,000 draws). Lower MSE is better;
higher physical-distance Spearman correlation is better. The fixed small
population and random candidate distribution limit extrapolation to CEM.}
\label{fig:development-action-ranking}\end{figure}
''')
    (REPORT/'summary.json').write_text(json.dumps({'status':'complete','source_sha256':sources,'registration_sha256':digest,'summaries':summary},indent=2)+'\n')
    print('Complete: 16 development cases, 528 actual candidate rollouts, 6336 model/candidate predictions')
if __name__=='__main__':main()
