#!/usr/bin/env python3
"""Source-checked complete reliability diagnostic table; no model computation."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"paper/generated/real_video"
SOURCE=ROOT/"reports/real_video_development/reliability_blend_results.json"
VERIFY=ROOT/"reports/real_video_development/reliability_blend_verification.json"
LABELS=(
    ("framewise","Framewise, calibrated"),
    ("factorized",r"ShiftWM (ours), calibrated"),
    ("equal_blend",r"Equal blend ($\alpha=0.5$)"),
    ("train_prior_blend","Global-prior blend (reference)"),
    ("train_query_blend","Train-query constant blend"),
    ("unshrunk_local","Unshrunk support gate"),
    ("shrunk_local",r"\textbf{Shrunk support gate (candidate)}"),
    ("one_step_local","One-step support gate"),
    ("shuffled_local","Shuffled support gate"),
    ("persistence","Persistence"),
    ("constant_velocity","Constant velocity"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result=json.loads(SOURCE.read_text());verification=json.loads(VERIFY.read_text())
    if (result['status']!='completed' or verification['status']!='passed'
            or verification['report_sha256']!=sha(SOURCE)
            or set(result['summary'])!={key for key,_ in LABELS}
            or [row['seed'] for row in result['runs']]!=[0,1,2]):
        raise ValueError('Require the complete independently verified study')
    registry=ROOT/'configs/real_video_development/reliability_blend_registration_v1.json'
    if sha(registry)!=result['registration_sha256']:
        raise ValueError('Registry changed')
    ledger=[]
    lines=[r'% Generated from the complete verified reliability development report.',
           r'\begin{table}[t]\centering\small',
           r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.08}',
           r'\begin{tabular}{lrrrr}\toprule',
           r'& \multicolumn{2}{c}{Endpoint MSE $\downarrow$} & \multicolumn{2}{c}{Gain vs. global prior $\uparrow$}\\',
           r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
           r'Method & h5 & h10 & h5 (\%) & h10 (\%)\\\midrule']
    for key,label in LABELS:
        item=result['summary'][key]
        gains={h:100*(1-item[h]/result['summary']['train_prior_blend'][h]) for h in ('h5','h10')}
        for h in ('h5','h10'):
            recomputed=sum(row['scores']['summary'][key][h] for row in result['runs'])/3
            if abs(recomputed-item[h])>1e-12:raise ValueError('Summary arithmetic differs')
        formatted={h:(r'\positivegain{'+f'{gains[h]:+.3f}'+r'}' if gains[h]>0 else f'{gains[h]:+.3f}') for h in ('h5','h10')}
        if key=='train_prior_blend':lines.append(r'\rowcolor{gray!8}')
        if key=='shrunk_local':lines.append(r'\rowcolor{orange!9}')
        lines.append(label+f" & {item['h5']:.6f} & {item['h10']:.6f} & "+formatted['h5']+' & '+formatted['h10']+r'\\')
        ledger.append({'arm':key,'h5':item['h5'],'h10':item['h10'],'gain_percent':gains,
                       'colored_cells':[h for h in gains if gains[h]>0]})
    primary=result['paired_intervals_vs_train_prior']['shrunk_local']['h5']
    primary_gain=100*(1-result['summary']['shrunk_local']['h5']/result['summary']['train_prior_blend']['h5'])
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{\textbf{A support-reliability candidate that did not pass its development gate.} '
        r'Original camera-1 validation only: 132 episodes, 57 sessions and 1,351 common windows, '
        r'equally weighted over episodes and three seeds. All arms have thirteen observed prefix frames '
        r'and identical ten-step query windows, whereas the main forecasting protocol uses three support frames. '
        r'Gains are relative to the \emph{global-prior blend}, which equals calibrated ShiftWM because '
        r'all training-fitted priors are one. \positivegain{Bold green} marks positive point estimates, '
        r'not significance or a passed decision rule. The candidate improves h5 by only '
        +f'{primary_gain:.3f}'+r'\%; its paired difference is $'
        +f"{primary['mean_difference']:+.8f}"+r'$, with 95\% interval $['
        +f"{primary['ci95'][0]:+.8f},{primary['ci95'][1]:+.8f}"+r']$. Ordinary equal averaging has the lowest mean errors. '
        r'The candidate fails the prespecified 1\% improvement, interval and all-seed criteria; '
        r'it is retained as a negative development result. No new neural weights are trained.}',
        r'\label{tab:reliability-development}\end{table}']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    table=OUTPUT/'reliability_development_table.tex';table.write_text('\n'.join(lines)+'\n')
    proof=OUTPUT/'reliability_development_proof.tex'
    proof.write_text(r'''\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage{booktabs,xcolor,colortbl,amsmath}
\definecolor{gainpositive}{HTML}{166534}
\newcommand{\positivegain}[1]{\textcolor{gainpositive}{\textbf{#1}}}
\pagestyle{empty}
\begin{document}
\typeout{RELIABILITY_TEXTWIDTH=\the\textwidth}
\input{generated/real_video/reliability_development_table.tex}
\end{document}
''')
    sources={str(path.relative_to(ROOT)):sha(path) for path in (SOURCE,VERIFY,registry,Path(__file__),table,proof,
             ROOT/'paper/sections/reliability_development.tex')}
    evidence={'status':'source_verified','sources':sources,'complete_arms':len(ledger),'rows':ledger,
              'reference':'train_prior_blend','positive_style':'positive relative point estimate only; not significance',
              'promotion':result['promote'],'primary_candidate_interval':result['paired_intervals_vs_train_prior']['shrunk_local']['h5'],
              'h10_candidate_interval':result['paired_intervals_vs_train_prior']['shrunk_local']['h10'],
              'precision':'FP64 blend sufficient-statistic MSE from FP32 donor forecasts; exported inference is FP32'}
    (OUTPUT/'reliability_development_sources.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'table':str(table.relative_to(ROOT)),'arms':len(ledger),'source_sha256':sha(SOURCE)}))


if __name__=='__main__':main()
