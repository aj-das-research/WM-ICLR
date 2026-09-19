#!/usr/bin/env python3
"""Current spatial architecture with right-hand mathematical contract.

New editorial source only. The original two-context/CEM diagram is not reused.
"""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.path import Path as MP
import numpy as np
from PIL import Image
from teaser_story_glyphs import feature_tensor,COBALT,TEAL,CORAL,GOLD
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/spatial_architecture_math'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE=ROOT/'paper/figure_sources/spatial_method/canonical_graph.json'
GRAPH=json.loads(BASE.read_text());EXPECTED=dict(GRAPH['sources_sha256'])
EXPECTED.update({'paper/figure_sources/teaser_gallery/asset_manifest.json': '5f323eacc1615e60f6b56603f66e6d8d84343aa2e66b87ff0db2a677e2e744cf', 'paper/figure_sources/teaser_gallery/assets/pusht_train_000011_frame0.png': 'd2e7cd6b8a60c2f8529a3aa00103433ba03596af327b304a0fc7876d4c5b79a1', 'paper/figures/split_assets/reacher.png': 'af1841dc653b987a38d6f660fed16bcac908840145e9fa64d2651efd5e724f2d'})
GLYPH_SHA='dc96d56f03423e831d8dc98b0229566373c916f4bf46ae3c42ac1a0284b3ea10'
INK='#26364B';MUTED='#657587';BORDER='#B9C6D2';NEUTRAL='#F5F7FA'
CAPTION=r'''\textbf{ShiftWM (ours): observed evidence, learned mixtures and bounded innovation.} (a) Three recorded DROID frames become frozen DINOv2 $4\!\times\!4$ features, normalized with fixed training-only channel statistics shared across patches. Spatial encoding feeds past-only patch-mean context and FiLM; a chronological action GRU conditions the temporal predictor. (b) All $H_h$ aliases denote the same horizon state. Fixed $Z_0\in\mathbb R^{16\times384}$ bypasses trainable encoding; keys instead use its last-observed encoding $E_0\in\mathbb R^{16\times96}$ before FiLM. Complementary sigmoid weights combine the fixed and mixed sources before adding the separate bounded innovation. (c) $Q,K$ are learned linear heads, $G$ is affine, and $R$ is LayerNorm plus an affine $96\to384$ projection. The envelope bounds standardized feature coordinates, not prediction error, with $m_c=\min_jZ_{0,jc}$ and $M_c=\max_jZ_{0,jc}$. Offline training uses $\mathcal L=(10B\cdot16\cdot384)^{-1}\sum_{b=1}^B\sum_{h=1}^{10}\|\hat Z_{bh}-Z_{bh}\|_F^2$; future references enter only this loss. The strip locates completed DROID forecasting, task-specific IWS decoder-transfer training~\citep{zhang2026rlawm}, and separately trained historical two-context planning, illustrated by an actual Reacher training frame. Only DINOv2 weights are frozen. Feature glyphs and bars are schematic, not generated RGB; DROID photographs (CC BY 4.0) and the attributed IWS training image are unchanged.'''


def drafts():
 DESIGN.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8})
 fig,axs=plt.subplots(1,3,figsize=(11,3),facecolor='white')
 plans=[('A  Visual hierarchy | aligned math',[(.08,.56,.53,.33,'observations + causal state'),(.08,.11,.53,.37,'source / gate / correction'),(.66,.11,.30,.78,'weights\nconvex mix\nbound\ntraining')]),
 ('B  Input rail + parallel contracts',[(.06,.75,.90,.15,'recorded observations and actions'),(.06,.10,.43,.57,'visual decoder'),(.55,.10,.41,.57,'equations + range')]),
 ('C  Equation-linked operation rows',[(.05,.67,.49,.20,'fixed observed source'),(.58,.67,.36,.20,'Q / K / T'),(.05,.39,.49,.20,'gate / mixture'),(.58,.39,.36,.20,'convex weights'),(.05,.11,.49,.20,'bounded correction'),(.58,.11,.36,.20,'envelope + loss')])]
 for ax,(title,regions) in zip(axs,plans):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.02,.99,title,va='top',fontsize=9,color=INK)
  for x,y,w,h,label in regions:
   ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.02',facecolor='#F1F7FA',edgecolor=BORDER,lw=.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8,color=INK)
 fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.pdf');fig.savefig(DESIGN/'composition_drafts.png',dpi=150);plt.close(fig)


def render():
 for p,h in EXPECTED.items():
  if sha(ROOT/p)!=h:raise ValueError('Frozen source changed: '+p)
 if sha(ROOT/'paper/scripts/teaser_story_glyphs.py')!=GLYPH_SHA:raise ValueError('Accepted glyph source changed')
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
 W,H=396,280.8;fig=plt.figure(figsize=(5.5,3.9),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
 texts=[];nodes={};edges=[];collapsed={}
 def text(x,y,label,ha='center',size=8,color=INK):
  color={TEAL:'#087F7D',CORAL:'#B64A45',GOLD:'#9B681B'}.get(color,color)
  t=ax.text(x,y,label,ha=ha,va='center',fontsize=size,color=color,zorder=10);texts.append(t);return t
 def box(name,x,y,w,h,label=None,color=BORDER,fill=NEUTRAL,dashed=False):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',facecolor=fill,edgecolor=color,lw=.65,ls='--' if dashed else '-',zorder=3));nodes[name]=[x,y,w,h]
  if label:text(x+w/2,y+h/2,label)
 def arrow(name,points,color=MUTED,semantic=None,control=False):
  ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=4.8,lw=.75,color=color,ls=(0,(2,1.4)) if control else '-',zorder=4));edges.append({'id':name,'semantic_id':semantic or name,'points':points,'control':control})
 def line(points,color=MUTED,lw=.6):ax.plot(*zip(*points),color=color,lw=lw,zorder=2)
 def op(name,x,y,symbol,r=5,color=TEAL):
  ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=color,lw=.85,zorder=5));text(x,y,symbol);nodes[name]=[x-r,y-r,2*r,2*r]
 def tensor(name,x,y,w=38,h=32,role='anchor'):
  o=feature_tensor(ax,x,y,w,h,role);nodes[name]=list(o['bounds']);return o
 def badge(x,y,n,color=TEAL):
  ax.add_patch(Circle((x,y),5.2,facecolor='white',edgecolor=color,lw=.75,zorder=6));text(x,y,str(n),size=8,color=color)
 text(5,273,'a  DROID instantiation: recorded history and causal actions',ha='left',size=8.5)
 main_ax=ax
 # Actual source images stay unchanged; source indices are stored-frame indices.
 for j,f in enumerate((0,5,10)):
  image=ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{f}.png'
  ia=fig.add_axes([(5+j*29)/W,244/H,27/W,15.1875/H]);ia.imshow(np.asarray(Image.open(image)),interpolation='none');ia.axis('off')
 text(46,234,'Frames 0 / 5 / 10',color=COBALT)
 box('frozen-dino',100,243,65,26,'Frozen DINO\npool + norm',COBALT,'#F2F5FF',True)
 arrow('observed-rgb-to-frozen-dino',[(92,256),(99,256)],COBALT)
 tensor('normalized-support',177,240,34,28,'observed');text(194,232,r'$Z_{-2:0}$',color=COBALT)
 arrow('frozen-dino-to-normalized-patch-grids',[(166,255),(176,255)],COBALT)
 box('spatial-encoder',223,243,59,26,'Spatial\nencoder')
 arrow('support-grids-to-spatial-encoder',[(212,254),(222,254)],COBALT)
 box('support-film',306,243,32,26,'FiLM')
 arrow('encoded-support-to-film',[(283,256),(305,256)],COBALT)
 box('past-context',223,199,59,23,'Past-only\ncontext')
 arrow('mean-encoded-support-to-transition-context',[(249,242),(249,223)],COBALT)
 text(257,233,'mean',ha='left',size=8,color=COBALT)
 box('past-actions',140,202,63,20,'2 past blocks',GOLD,'#FFF9EC')
 arrow('past-actions-to-transition-context',[(204,212),(222,212)],GOLD)
 arrow('context-to-film',[(283,210),(322,210),(322,242)],TEAL)
 box('lewm-temporal',350,243,43,26,'Temporal\npredictor')
 arrow('conditioned-support-to-lewm',[(339,258),(349,258)],COBALT)
 box('future-action-prefix',140,175,63,19,'Query prefix',GOLD,'#FFF9EC')
 box('action-prefix-gru',223,175,59,19,'Prefix GRU')
 arrow('future-prefix-to-causal-gru',[(204,184),(222,184)],GOLD)
 arrow('past-actions-to-causal-gru',[(173,201),(173,198),(214,198),(214,188),(222,188)],GOLD)
 arrow('causal-action-states-to-lewm',[(283,184),(343,184),(343,249),(349,249)],GOLD)
 box('horizon-state',353,207,40,17,r'$H_h$',TEAL,'#EEF9F7')
 arrow('lewm-to-horizon-state',[(370,242),(370,225)],TEAL)
 # Separate evidence-source aliases: raw normalized Z0 and pre-FiLM E0.
 text(6,214,'Fixed evidence',ha='left',size=8.3,color=COBALT)
 text(6,200,r'$Z_0$: last normalized grid',ha='left',size=8)
 text(6,187,r'$E_0$: last encoding, pre-FiLM',ha='left',size=8)
 text(6,174,'Same source at every horizon',ha='left',size=8,color=MUTED)
 line([(5,166),(393,166)],'#DCE4EB',.65)
 text(5,158,'b  Graphic decoder',ha='left',size=8.5)
 text(244,158,'c  Exact decoder',ha='left',size=8.5)
 line([(235,27),(235,162)],'#DCE4EB',.65)
 # Decoder uses its own translated point-coordinate frame (origin y=24pt).
 ax=fig.add_axes([0,24/H,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
 # Named fixed sources and horizon aliases keep causal preparation and decoder
 # readable without falsely joining the source Z0 and encoded E0 branches.
 anchor=tensor('fixed-anchor',7,77,38,32,'anchor');text(26,68,r'Fixed $Z_0$',color=COBALT)
 box('query-alias',78,111,25,16,r'$H_h$',TEAL,'#EEF9F7')
 box('key-alias',129,111,25,16,r'$E_0$',COBALT,'#F2F5FF')
 box('mix-and-product',78,86,78,20,color=TEAL,fill='#EEF9F7')
 T=.75*np.eye(16)+.2*np.roll(np.eye(16),1,axis=1)+.05*np.ones((16,16))/16
 for rr in range(16):
  for cc in range(16):
   value=T[rr,cc]/T.max();rgb=np.asarray(matplotlib.colors.to_rgb(TEAL));fill=tuple(1-value*.92*(1-rgb))
   ax.add_patch(Rectangle((83+cc,88+15-rr),1,1,facecolor=fill,edgecolor='white',lw=.025,zorder=4))
 text(124,96,r'$T_h\,Z_0$',color=TEAL)
 arrow('horizon-state-to-query',[(92,110),(92,106)],TEAL)
 arrow('pre-film-observation-to-key',[(140,110),(140,106)],COBALT)
 arrow('fixed-anchor-to-feature-product',[(46,93),(77,93)],COBALT)
 mixed=tensor('mixed-anchor',184,77,38,32,'mixed');text(203,68,r'$T_hZ_0$',color=TEAL)
 arrow('product-to-mixed-anchor',[(157,91),(183,91)],TEAL)
 source_colors=np.asarray([p.get_facecolor()[:3] for p in anchor['artists'][4:20]])
 for piece,color in zip(mixed['artists'][4:20],T@source_colors):piece.set_facecolor(color)
 collapsed['mix-and-product']=['queries-to-transport-scores','keys-to-transport-scores','softmax-to-row-stochastic-matrix','transport-to-feature-product']
 # The per-patch gate controls two distinct contributions, not a third summand.
 box('patch-gate',81,52,71,26,color=CORAL,fill='#FFF5F1');text(116,69,r'$g_h=\sigma(G(H_h))$',color=CORAL)
 for j,v in enumerate((.25,.58,.72,.42)):
  x=87+j*16;ax.add_patch(Rectangle((x,56),12,4,facecolor=COBALT,edgecolor='none',zorder=4));ax.add_patch(Rectangle((x+12*(1-v),56),12*v,4,facecolor=TEAL,edgecolor='none',zorder=5))
 collapsed['patch-gate']=['horizon-state-to-gate']
 op('fixed-coefficient',61,40,'×',r=4.5,color=COBALT);op('mixed-coefficient',174,40,'×',r=4.5,color=TEAL);op('complementary-blend',117,40,r'$\Sigma$',r=5.5,color=CORAL)
 arrow('identity-anchor-to-complementary-blend',[(6,93),(2,93),(2,40),(55,40)],COBALT)
 arrow('weighted-identity-to-sum',[(66,40),(110,40)],COBALT,'identity-anchor-to-complementary-blend')
 arrow('mixed-anchor-to-complementary-blend',[(223,93),(230,93),(230,40),(180,40)],TEAL)
 arrow('weighted-mixture-to-sum',[(168,40),(124,40)],TEAL,'mixed-anchor-to-complementary-blend')
 arrow('gate-to-complementary-blend',[(94,51),(94,50),(61,50),(61,46)],CORAL,control=True)
 arrow('gate-to-mixture-weight',[(138,51),(138,50),(174,50),(174,46)],CORAL,'gate-to-complementary-blend',True)
 text(61,30,r'$1-g_h$',color=COBALT);text(174,30,r'$g_h$',color=TEAL)
 # A compact projection/range cue is separate from the convex mixture.
 box('bounded-innovation',5,3,91,22,color=GOLD,fill='#FFFAEE');text(50,19,r'$\Delta_h=\tanh R(H_h)$',size=8,color=GOLD)
 line([(25,8),(76,8)],GOLD,.6)
 for x in (25,76):line([(x,5),(x,11)],GOLD,.75)
 text(16,8,r'$-1$',size=8,color=MUTED);text(86,8,r'$+1$',size=8,color=MUTED)
 ax.add_patch(Circle((48,8),1.4,facecolor=GOLD,edgecolor='white',lw=.25,zorder=5))
 collapsed['bounded-innovation']=['horizon-state-to-projected-innovation']
 op('addition',159,15,'+',r=5,color=GOLD)
 arrow('blend-to-addition',[(117,34),(117,15),(153,15)],TEAL)
 arrow('bounded-innovation-to-addition',[(97,12),(144,12),(144,4),(159,4),(159,9)],GOLD)
 tensor('forecast',189,2,34,28,'forecast');text(179,19,r'$\hat Z_h$',size=8,color=TEAL)
 arrow('addition-to-future-features',[(165,15),(188,15)],TEAL)
 # Right column equations define exactly the visual operators; loss is captioned.
 ax=main_ax
 def mathline(y,formula,size=8):text(244,y,formula,ha='left',size=size)
 mathline(144,r'$S_h=Q(H_h)K(E_0)^\top/\sqrt{96}+4I$')
 mathline(131,r'$T_h=\mathrm{softmax}_{\rm row}(S_h)$')
 mathline(118,r'$g_h=\sigma(G(H_h)),\quad D_h=\mathrm{diag}(g_h)$',size=8)
 line([(244,108),(390,108)],'#E2E8EE')
 mathline(96,r'$W_h=I-D_h+D_hT_h$')
 mathline(82,r'$\hat Z_h=W_hZ_0+\Delta_h$')
 mathline(68,r'$W_h\geq0,\qquad W_h\mathbf{1}=\mathbf{1}$')
 line([(244,57),(390,57)],'#E2E8EE')
 mathline(44,r'$m_c-1\leq\hat Z_{h,ic}\leq M_c+1$')
 text(244,31,'Normalized channel envelope',ha='left',size=8,color=MUTED)
 # Task strip is a study locator, never a dataflow from this decoder to a planner.
 line([(5,23),(393,23)],'#DCE4EB',.65)
 applications=[
  ('DROID forecasting','completed',ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png',5,34,19.125),
  ('IWS decoder transfer','training · 3 tasks',ROOT/'paper/figure_sources/teaser_gallery/assets/pusht_train_000011_frame0.png',139,28,21),
  ('Simulation planning','historical models',ROOT/'paper/figures/split_assets/reacher.png',272,21,21)]
 task_assets=[]
 for title,status,path,x,w,h in applications:
  ia=fig.add_axes([x/W,1/H,w/W,h/H]);ia.imshow(np.asarray(Image.open(path)),interpolation='none');ia.axis('off')
  text(x+w+5,16,title,ha='left',size=8);text(x+w+5,6,status,ha='left',size=8,color=MUTED)
  task_assets.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'crop':'none','title':title,'status':status})
 # Preserve all canonical semantic dependencies, including documented collapsed internals.
 expected={e['semantic_id'] for e in GRAPH['semantic_edges']}
 actual={e['semantic_id'] for e in edges}|{s for group in collapsed.values() for s in group}
 assert actual==expected,(expected-actual,actual-expected)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes=[t.get_window_extent(renderer) for t in texts]
 overlaps=[[texts[i].get_text(),texts[j].get_text()] for i,b in enumerate(boxes) for j,c in enumerate(boxes) if i<j and b.overlaps(c)]
 clipped=[t.get_text() for t,b in zip(texts,boxes) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1]
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'spatial_architecture_math.{ext}',dpi=300)
 plt.close(fig)
 include='\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_architecture_math.pdf}\n\\caption[Spatial decoder and mathematical contract.]{'+CAPTION+'}\n\\label{fig:editorial-spatial-method}\n\\end{figure}\n'
 (OUT/'spatial_architecture_math_caption.tex').write_text(CAPTION+'\n');(OUT/'spatial_architecture_math_figure.tex').write_text(include)
 graph={'canonical_semantic_dependencies':sorted(expected),'rendered_edges':edges,'collapsed_subgraphs':collapsed,
        'alias_contract':'All H_h aliases same horizon state; E0 is fixed last pre-FiLM encoding, Z0 fixed last standardized observed grid. Source Z0 bypasses trainable encoding.',
        'right_math_is_definition_not_extra_model_inputs':True,'collapse_precision':{'mix-and-product':'Functional query/key heads, score scaling plus4I, row softmax, then multiply fixed normalizedZ0; exact formula at right.','patch-gate':'Per-patch affine sigmoid head on the sameHh alias.','bounded-innovation':'LayerNorm+Linear96to384 followed by tanh, represented by exact expression and range cue.'},'historical_two_context_CEM_or_paired_losses':False,'nodes':nodes,'decoder_coordinate_origin_points':[0,24],'task_strip_is_not_model_dataflow':True}
 (DESIGN/'canonical_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
 evidence={'status':'candidate_pending_pixel_and_mathematical_review','renderer_sha256':sha(__file__),'sources_sha256':EXPECTED,
           'task_strip_assets':task_assets,'glyph_sha256':GLYPH_SHA,'canonical_original_graph_sha256':sha(BASE),'canonical_new_graph_sha256':sha(DESIGN/'canonical_graph.json'),
           'dimensions_inches':[5.5,3.9],'minimum_font_pt':8,'semantic_dependencies_preserved':len(actual),'text_overlaps':overlaps,'clipped':clipped,
           'outputs_sha256':{f'spatial_architecture_math.{x}':sha(OUT/f'spatial_architecture_math.{x}') for x in ('pdf','svg','png')},
           'illustration_boundary':'Feature glyphs, gate bars and bound ticks are schematic. Actual three source images unchanged. No numerical results or RGB forecasts.'}
 (OUT/'spatial_architecture_math_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps({'semantic_dependencies':len(actual),'overlaps':overlaps,'clipped':clipped,'integrated':False}))

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--drafts-only',action='store_true');a=p.parse_args();drafts() if a.drafts_only else render()
