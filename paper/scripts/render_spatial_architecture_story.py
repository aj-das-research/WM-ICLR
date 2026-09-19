#!/usr/bin/env python3
"""Visual spatial decoder story; preserves the reviewed 28-dependency graph."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.path import Path as MP
from PIL import Image,ImageOps
from teaser_story_glyphs import feature_tensor,COBALT,TEAL,CORAL,GOLD
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/spatial_architecture_story'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE=ROOT/'paper/design/spatial_architecture_refined/canonical_graph.json'
GRAPH=json.loads(BASE.read_text());EXPECTED=GRAPH['sources_sha256']
GLYPH_SHA='dc96d56f03423e831d8dc98b0229566373c916f4bf46ae3c42ac1a0284b3ea10'
INK='#26364B';MUTED='#657587';BORDER='#B9C6D2';NEUTRAL='#F5F7FA'
CAPTION=r'''\textbf{ShiftWM (ours): observed features remain the source of every forecast.} (a) Frozen DINOv2 features are pooled to $4\!\times\!4$ and normalized using fixed training-only channel statistics. Trainable spatial encoding feeds a past-only patch-mean context and FiLM. A separate chronological action GRU supplies the two past states and the prefix endpoint at $h$ to the trained LeWM temporal architecture. (b) Every $H_h$ port denotes the same horizon state. Keys use the fixed last-observed spatial encoding $E_0$ before FiLM; the source $Z_0$ is instead the last observed normalized $16\!\times\!384$ feature grid. Dashed coral control paths apply complementary per-patch weights to the two feature branches. Row-stochastic mixing and this sigmoid gate form $(1-g_h)Z_0+g_hT_hZ_0$. The distinct innovation branch applies $R=\mathrm{LayerNorm}+\mathrm{Linear}_{96\to384}$ and $\Delta_h=\tanh R(H_h)$, then adds it to the blend. The unit bound applies in normalized coordinates; inverse normalization is omitted. Only DINOv2 weights are frozen. Colored feature objects, gate bars and saturation examples are schematic; DROID observations are unchanged (CC BY 4.0). No future RGB enters the model. Appendix~\ref{app:spatial-development} gives the full protocol.'''


def drafts():
 DESIGN.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8})
 fig,axs=plt.subplots(1,3,figsize=(12,3.2),facecolor='white')
 plans=[('A · State beside visual decoder',[(.16,.81,'observed'),(.16,.56,'context/actions'),(.16,.27,'H'),(.59,.85,'Q / K → T'),(.49,.57,'fixed source'),(.82,.57,'mixed source'),(.65,.32,'gate blend'),(.65,.10,'bound + output')],[(0,1),(1,2),(3,5),(4,5),(4,6),(5,6),(6,7)]),('B · State rail above forked decoder',[(.14,.84,'observed'),(.49,.84,'context/actions'),(.85,.84,'H'),(.15,.48,'fixed source'),(.48,.48,'mix + gate'),(.81,.48,'projection'),(.51,.15,'blend + bound')],[(0,1),(1,2),(2,4),(2,5),(3,4),(4,6),(5,6)]),('C · Central anchor with decoder loop',[(.20,.84,'observed'),(.71,.84,'causal H'),(.48,.55,'fixed source'),(.18,.32,'mix weights'),(.78,.32,'gate / bound'),(.49,.10,'forecast')],[(0,1),(0,2),(1,3),(1,4),(2,3),(3,5),(4,5)])]
 for ax,(title,nodes,edges) in zip(axs,plans):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.01,.97,title,va='top',fontsize=9,color=INK)
  for a,b in edges:
   x,y,_=nodes[a];xx,yy,_=nodes[b];ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=7,shrinkA=18,shrinkB=20,lw=.7,color=MUTED))
  for x,y,label in nodes:
   ax.add_patch(FancyBboxPatch((x-.115,y-.05),.23,.10,boxstyle='round,pad=0,rounding_size=.01',facecolor=NEUTRAL,edgecolor=BORDER,lw=.6));ax.text(x,y,label,ha='center',va='center',fontsize=8,color=INK)
 fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.pdf');fig.savefig(DESIGN/'composition_drafts.png',dpi=150);plt.close(fig)


def render(integrate=False):
 for p,h in EXPECTED.items():
  if sha(ROOT/p)!=h:raise ValueError('Scientific source changed: '+p)
 if sha(ROOT/'paper/scripts/teaser_story_glyphs.py')!=GLYPH_SHA:raise ValueError('Accepted teaser glyph module changed')
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
 W,H=396,280.8;fig=plt.figure(figsize=(5.5,3.9),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
 texts=[];nodes={};edges=[]
 def text(x,y,label,ha='center',size=8,color=INK):
  label_color={TEAL:'#087F7D',CORAL:'#B64A45',GOLD:'#9B681B'}.get(color,color)
  t=ax.text(x,y,label,ha=ha,va='center',fontsize=size,color=label_color,zorder=10);texts.append(t);return t
 def box(name,x,y,w,h,label=None,color=BORDER,fill=NEUTRAL,dashed=False):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',facecolor=fill,edgecolor=color,lw=.65,linestyle='--' if dashed else '-',zorder=3));nodes[name]=[x,y,w,h]
  if label:text(x+w/2,y+h/2,label,color=INK)
 def arrow(name,points,color=MUTED,semantic=None,control=False):
  ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=5,linewidth=.75,color=color,linestyle=(0,(2,1.4)) if control else '-',zorder=4));edges.append({'id':name,'semantic_id':semantic or name,'points':points,'color':color,'control':control})
 def line(points,color=MUTED,lw=.6):ax.plot(*zip(*points),color=color,lw=lw,zorder=2)
 def operator(name,x,y,symbol,r=6,color=TEAL):
  ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=color,lw=.85,zorder=5));text(x,y,symbol);nodes[name]=[x-r,y-r,2*r,2*r]
 def tensor(name,x,y,w=38,h=32,role='anchor'):
  obj=feature_tensor(ax,x,y,w,h,role);nodes[name]=list(obj['bounds']);return obj
 text(8,274,'a  Causal observed history',ha='left',size=8.7)
 text(164,274,'b  Anchor, mix and bound',ha='left',size=8.7)
 line([(153,6),(153,266)],'#E0E6EB',.55)
 # Actual unchanged source observations; the feature objects remain illustrative.
 for j,f in enumerate((0,5,10)):
  p=ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{f}.png';ia=fig.add_axes([(9+j*46)/W,239/H,40/W,22.5/H]);ia.imshow(np.asarray(Image.open(p)),interpolation='none');ia.set_axis_off()
 line([(10,236),(136,236)],COBALT);arrow('observed-rgb-to-frozen-dino',[(77,236),(77,229)],COBALT)
 box('frozen-dino',32,206,91,22,'Frozen DINOv2',COBALT,'#F2F5FF',True)
 arrow('frozen-dino-to-normalized-patch-grids',[(77,205),(77,195)],COBALT)
 line([(26,195),(116,195)],COBALT)
 for j in range(3):tensor('observed-grid-'+str(j),9+j*44,163,36,30,'observed')
 line([(10,160),(133,160)],COBALT);text(43,155,r'$Z_{-2},Z_{-1},Z_0$',color=COBALT)
 box('spatial-encoder',83,122,62,23,'Spatial\nencoder')
 arrow('support-grids-to-spatial-encoder',[(114,160),(114,146)],COBALT)
 box('past-actions',8,132,57,18,r'Past $a_{-1:0}$',GOLD,'#FFF9EC')
 box('past-context',8,95,56,24,'Past-only\ncontext')
 arrow('past-actions-to-transition-context',[(14,131),(14,120)],GOLD)
 arrow('mean-encoded-support-to-transition-context',[(82,133),(74,133),(74,110),(65,110)],COBALT)
 box('support-film',83,83,62,23,'FiLM')
 arrow('encoded-support-to-film',[(114,121),(114,107)],COBALT)
 arrow('context-to-film',[(65,100),(78,100),(78,94),(82,94)],TEAL)
 box('future-action-prefix',8,71,56,17,r'Prefix $a_{1:h}$',GOLD,'#FFF9EC')
 box('action-prefix-gru',8,39,56,24,'Prefix GRU')
 arrow('past-actions-to-causal-gru',[(7,141),(3.5,141),(3.5,69),(23,69),(23,64)],GOLD)
 arrow('future-prefix-to-causal-gru',[(45,70),(45,64)],GOLD)
 box('lewm-temporal',83,39,62,24,'Temporal\npredictor')
 arrow('conditioned-support-to-lewm',[(114,82),(114,64)],COBALT)
 arrow('causal-action-states-to-lewm',[(65,51),(82,51)],GOLD)
 box('horizon-state',83,8,62,18,r'$H_h$: 16 × 96',TEAL,'#EEF9F7')
 arrow('lewm-to-horizon-state',[(114,38),(114,27)],TEAL)
 text(39,19,'Dashed box:\nfrozen',color=MUTED)
 # Repeated named tensor ports refer to precisely the same source states.
 box('query-state-alias',177,249,30,17,r'$H_h$',TEAL,'#EEF9F7')
 box('pre-film-key-port',318,249,30,17,r'$E_0$',COBALT,'#F2F5FF');text(351,257,'pre-FiLM',ha='left',color=COBALT)
 text(198,237,r'$Q_h=W_QH_h$',color=TEAL);text(333,237,r'$K_0=W_KE_0$',color=COBALT)
 arrow('horizon-state-to-query',[(192,248),(192,243)],TEAL)
 arrow('pre-film-observation-to-key',[(333,248),(333,243)],COBALT)
 box('row-softmax',164,209,224,20,color=TEAL,fill='#F1FAF8')
 text(276,219,r'$T_h=\mathrm{softmax}_{\rm row}(Q_hK_0^\top/\sqrt{96}+4I)$',color=TEAL)
 arrow('queries-to-transport-scores',[(198,232),(198,230)],TEAL)
 arrow('keys-to-transport-scores',[(333,232),(333,230)],COBALT)
 T=.75*np.eye(16)+.2*np.roll(np.eye(16),1,axis=1)+.05*np.ones((16,16))/16
 for r in range(16):
  for c in range(16):
   value=T[r,c]/T.max();rgb=np.asarray(matplotlib.colors.to_rgb(TEAL));fill=tuple(1-value*.92*(1-rgb));ax.add_patch(Rectangle((178+c*28/16,173+(15-r)*28/16),28/16,28/16,facecolor=fill,edgecolor='white',lw=.04,zorder=3))
 ax.add_patch(Rectangle((178,173),28,28,facecolor='none',edgecolor=TEAL,lw=.6,zorder=4));text(191,168,r'$T_h$',color=TEAL)
 arrow('softmax-to-row-stochastic-matrix',[(191,208),(191,202)],TEAL)
 anchor_object=tensor('fixed-last-observed-anchor',164,124);source_colors=np.asarray([p.get_facecolor()[:3] for p in anchor_object['artists'][4:20]]);text(183,114,r'Fixed $Z_0$',color=COBALT)
 operator('matrix-times-anchor',225,187,'×')
 arrow('fixed-anchor-to-feature-product',[(183,157),(183,161),(225,161),(225,180)],COBALT)
 arrow('transport-to-feature-product',[(207,187),(218,187)],TEAL)
 mixed_object=tensor('mixed-anchor',345,124,role='mixed')
 for piece,color in zip(mixed_object['artists'][4:20],T@source_colors):piece.set_facecolor(color)
 text(364,114,r'$T_hZ_0$',color=TEAL)
 arrow('product-to-mixed-anchor',[(232,187),(364,187),(364,157)],TEAL)
 box('gate-state-alias',252,166,30,17,r'$H_h$',TEAL,'#EEF9F7')
 box('patch-sigmoid-gate',222,123,92,36,color=CORAL,fill='#FFF5F1')
 text(268,152,'Patch gate',color=CORAL)
 text(268,139,r'$g_h=\sigma(W_gH_h+b_g)$',color=INK)
 for j,v in enumerate((.25,.58,.72,.42)):
  xx=231+j*19;ax.add_patch(FancyBboxPatch((xx,128),15,4,boxstyle='round,pad=0,rounding_size=1',facecolor=COBALT,edgecolor='none',zorder=4));ax.add_patch(Rectangle((xx+15*(1-v),128),15*v,4,facecolor=TEAL,edgecolor='none',zorder=5))
 arrow('horizon-state-to-gate',[(267,165),(267,160)],TEAL)
 operator('complementary-anchor-blend',268,96,r'$\Sigma$',r=7,color=CORAL)
 operator('complementary-source-weight',208,96,'×',r=5,color=COBALT)
 operator('mixed-source-weight',333,96,'×',r=5,color=TEAL)
 arrow('identity-anchor-to-complementary-blend',[(163,140),(158,140),(158,96),(202,96)],COBALT)
 arrow('weighted-identity-to-sum',[(214,96),(260,96)],COBALT,semantic='identity-anchor-to-complementary-blend')
 arrow('mixed-anchor-to-complementary-blend',[(384,140),(391,140),(391,96),(339,96)],TEAL)
 arrow('weighted-mixture-to-sum',[(327,96),(276,96)],TEAL,semantic='mixed-anchor-to-complementary-blend')
 text(208,84,r'$1-g_h$',color=COBALT);text(333,84,r'$g_h$',color=TEAL)
 arrow('gate-to-complementary-blend',[(268,122),(268,118),(208,118),(208,102)],CORAL,control=True)
 arrow('gate-to-mixed-source-weight',[(268,118),(333,118),(333,102)],CORAL,semantic='gate-to-complementary-blend',control=True)
 ax.add_patch(Circle((268,118),.9,facecolor=CORAL,edgecolor='none',zorder=5))
 operator('add-bounded-innovation',307,60,'+',color=GOLD)
 arrow('blend-to-addition',[(268,88),(268,60),(300,60)],TEAL)
 forecast_object=tensor('future-feature-forecast',346,43,38,32,'forecast')
 schematic_g=np.tile(np.array([.25,.58,.72,.42]),4)[:,None]
 schematic_delta=np.linspace(-.035,.035,16)[:,None]*np.array([[.6,-.4,.2]])
 schematic_forecast=(1-schematic_g)*source_colors+schematic_g*(T@source_colors)+schematic_delta
 for piece,color in zip(forecast_object['artists'][4:20],schematic_forecast):piece.set_facecolor(color)
 text(365,33,r'$\hat Z_h$',color=TEAL)
 arrow('addition-to-future-features',[(314,60),(345,60)],TEAL)
 # Small separate feature projection and saturation/range cue, not a tanh plot.
 box('innovation-state-alias',161,13,24,17,r'$H_h$',TEAL,'#EEF9F7')
 box('innovation-projection',193,11,24,21,r'$R$',color=CORAL,fill='#FFF5F1')
 arrow('horizon-state-to-projected-innovation',[(186,21.5),(192,21.5)],TEAL)
 raw=np.array([-1.8,.45,1.1]);bounded=np.tanh(raw)
 line([(232,11),(232,31)],'#E8C3BC',.5)
 for yy,value in zip((15,21,27),raw):
  xx=232+value*4;line([(232,yy),(xx,yy)],CORAL,1.0);ax.add_patch(Circle((xx,yy),1.05,facecolor=CORAL,edgecolor='none',zorder=4))
 arrow('projection-to-feature-piece',[(218,21.5),(224,21.5)],CORAL,semantic='horizon-state-to-projected-innovation')
 arrow('feature-piece-to-tanh',[(239,21.5),(259,21.5)],CORAL,semantic='horizon-state-to-projected-innovation')
 text(205,40,'project',color=CORAL);text(283,38,'tanh bound',color=GOLD)
 line([(264,11),(264,31)],GOLD,.8);line([(320,11),(320,31)],GOLD,.8)
 for yy,value in zip((15,21,27),bounded):
  line([(264,yy),(320,yy)],'#F6E3BA',.55);ax.add_patch(Circle((292+28*value,yy),1.4,facecolor=GOLD,edgecolor='white',lw=.25,zorder=5))
 text(264,5,r'$-1$',color=MUTED);text(320,5,r'$+1$',color=MUTED);text(294,5,r'$\Delta_h$',color=GOLD)
 arrow('bounded-innovation-to-addition',[(321,21),(330,21),(330,45),(307,45),(307,53)],GOLD)
 # Audit text geometry and preserve the semantic ledger, including split visual edges.
 fig.canvas.draw();ren=fig.canvas.get_renderer();bounds=[t.get_window_extent(ren) for t in texts];overlaps=[]
 for i,b in enumerate(bounds):
  for j,c in enumerate(bounds[i+1:],i+1):
   if b.overlaps(c):overlaps.append([texts[i].get_text(),texts[j].get_text()])
 clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.x1>fig.bbox.x1 or b.y0<0 or b.y1>fig.bbox.y1]
 original=json.loads((ROOT/'paper/generated/editorial/spatial_architecture_refined_evidence.json').read_text());expected_edges={e['id'] for e in original['arrows']};actual_edges={e['semantic_id'] for e in edges}
 assert expected_edges==actual_edges,(expected_edges-actual_edges,actual_edges-expected_edges)
 assert np.all(abs(bounded)<=1) and np.allclose(T.sum(1),1)
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'spatial_architecture_story.{ext}',dpi=300)
 fig.savefig(DESIGN/'paper_width.png',dpi=110);plt.close(fig);ImageOps.grayscale(Image.open(DESIGN/'paper_width.png')).save(DESIGN/'grayscale.png')
 evidence={'status':'candidate_pending_actual_review','sources_sha256':EXPECTED,'renderer_sha256':sha(__file__),'glyph_module_sha256':GLYPH_SHA,'canonical_graph_sha256':sha(BASE),'dimensions_inches':[5.5,3.9],'minimum_label_font_pt':8,'semantic_dependencies_preserved':28,'rendered_edges':edges,'nodes':nodes,'text_overlaps':overlaps,'clipped':clipped,'illustrative_only':{'mixing_matrix_row_sum_max_error':float(abs(T.sum(1)-1).max()),'schematic_color_gate':schematic_g.ravel().tolist(),'schematic_color_correction':schematic_delta.tolist(),'projection_example':raw.tolist(),'tanh_example':bounded.tolist(),'meaning':'Schematic channels, matrix and gates; never measured model values, RGB patches or physical displacement.'}}
 (OUT/'spatial_architecture_story_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');(OUT/'spatial_architecture_story_caption.tex').write_text(CAPTION+'\n')
 include='\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_architecture_story.pdf}\n\\caption[Observation-anchored spatial world model.]{'+CAPTION+'}\n\\label{fig:editorial-spatial-method}\n\\end{figure}\n'
 (OUT/'spatial_architecture_story_figure.tex').write_text(include)
 if integrate:(OUT/'spatial_architecture_figure.tex').write_text(include)
 print(json.dumps({'semantic_dependencies':len(actual_edges),'text_overlaps':overlaps,'clipped':clipped,'integrated':integrate}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--drafts-only',action='store_true');p.add_argument('--integrate',action='store_true');a=p.parse_args();drafts() if a.drafts_only else render(a.integrate)
