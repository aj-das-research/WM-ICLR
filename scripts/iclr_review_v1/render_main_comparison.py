"""Promote complete multi-metric/external-baseline evidence to the main paper."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/iclr_review_v1'
source=ROOT/'paper/generated/metric_completion_droid/scores.tex'
text=source.read_text().replace(r'\label{tab:droid-complementary-metrics}',r'\label{tab:editorial-spatial}\label{tab:droid-complementary-metrics}')
text=text.replace('DROID complementary errors at query step ten.','DROID development forecasting: internal controls and adapted published-model baselines.')
text=text.replace('All 33 learned','Query-step-ten endpoint errors. All 33 learned')
text=text.replace('External rows are\nadapted recipes,','DINO-WM rows use the official predictor with the adapted pooled-token interface \\citep{zhou2024dinowm}. They are\nadapted recipes,')
(OUT/'main_droid_table.tex').write_text(text)
section=ROOT/'paper/generated/metric_completion_droid/section.tex'
(OUT/'droid_complementary_section.tex').write_text(section.read_text().replace(r'\input{generated/metric_completion_droid/scores.tex}',r'Table~\ref{tab:droid-complementary-metrics} in the main paper reports all internal controls, persistence, and four adapted official DINO-WM recipes across the four endpoint metrics.'))
(OUT/'main_comparison_sources.json').write_text(json.dumps({'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [source,section]},'scope':'complete existing measurements promoted; no new scores'},indent=2)+'\n')
