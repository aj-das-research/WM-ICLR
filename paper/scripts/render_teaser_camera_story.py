#!/usr/bin/env python3
"""Picture-led DROID teaser with all nine scoped paper inputs.

New presentation only. Full recorded frames, frozen ledgers and model evidence
are never modified. The forecast glyphs describe features, never generated RGB.
"""
from pathlib import Path
import importlib.util,json,argparse,hashlib,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('camera_story_base',ROOT/'paper/scripts/render_teaser_benchmarks.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
assert B.sha(ROOT/'paper/scripts/render_teaser_benchmarks.py')=='55935cf3f7131e00a772618102adeb29b5254dfd13a66502ffda286e72f964b5'
DESIGN=ROOT/'paper/design/teaser_camera_story';PUBLIC=ROOT/'paper/figure_sources/teaser_camera_story';OUT=ROOT/'paper/generated/editorial'
INK='#1C304D';BLUE='#2378B7';TEAL='#008E88';CORAL='#D96951';GOLD='#CF942D';MUTED='#64748B'

CAPTION=(r'\textbf{Forecast from retained observations.} '
 r'Three recorded DROID frames supply fixed support features and the last-observed reference $Z_0$; the palette compresses this interface (Figure~\ref{fig:editorial-spatial-method}). '
 r'ShiftWM conditions every query on this support and supplied causal commands; autoregression rolls its own predictions into the next feature history, keeping inferred context fixed. '
 r'Two schematic endpoints illustrate feature forecasts, not generated RGB; the camera is illustrative. '
 r'Historical simulations use distinct models; IWS tests a separately trained single-observation adapter; Open-H is a physical-phantom input audit. '
 r'All 141 DROID development episodes remain visible. The 5.30\% reduction compares population mean native, training-standardized $h=10$ feature MSE with matched autoregression; uncertainty: Figure~\ref{fig:editorial-spatial}. '
 r'Images: DROID/Open-H (CC BY 4.0); IWS \citep{zhang2026rlawm}.')

def inventory():
 B.checked_gallery();return {r['id']:r for r in json.loads((ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json').read_text())['records']}

def photo(c,row,x,y,w):
 return c.photo(ROOT/row['asset'],x,y,w)

def camera(c,row,x,y,w):
 h=w*9/16
 # Rounded exterior/frame, rectangular unchanged experimental image inside.
 c.rect(x-7,y-12,w+15,h+22,'#EEF4F8','#8BA9BC',r=6,lw=.8)
 c.rect(x+9,y-17,25,6,'#C5D8E6',r=2)
 c.rect(x+w-12,y-16,13,4,BLUE,r=1.5)
 c.circle(x+w+3,y+h/2,2.2,BLUE)
 photo(c,row,x,y,w)
 c.text(x+w/2,y+h+11,'DROID · recorded observations',8,BLUE,'center')
 # No duplicate RGB or imagined frame is used to suggest a video outcome.

def token(c,x,y,w=22,col=TEAL,pred=True):
 # All forecast/reference glyphs share a4×4 spatial layout; schematic values.
 c.rect(x+2,y+2,w,w,'#E2E8EC',r=2,alpha=.7)
 c.rect(x,y,w,w,'white',col,r=2,lw=.7)
 for i in range(4):
  for j in range(4):c.rect(x+1+j*(w-2)/4,y+1+i*(w-2)/4,(w-2)/4-.5,(w-2)/4-.5,B.lighten(col,.2+.5*((i+j*2)%6)/6),r=.4)
 if pred:c.line([(x-2,y-2),(x+w+2,y-2),(x+w+2,y+w+2),(x-2,y+w+2),(x-2,y-2)],col,.6,dashed=True)

def cmd(c,x,y,port):
 c.circle(x,y,2,GOLD);c.line([(x,y+3),port],GOLD,.8,True,semantic='Available causal action prefix conditions this method')

def strip(c,inv,y=133):
 c.text(7,y,'Historical simulations',8,MUTED)
 c.text(183,y,'IWS · training inputs',8,MUTED)
 c.text(329,y,'Input audit',8,MUTED)
 names=[('sim_pusht','PushT',7,34),('sim_reacher','Reacher',49,34),('sim_drone','Drone',91,34),('sim_surgery','Tissue',133,34),('iws_pusht','PushT',183,39),('iws_bimanual_box','Box',230,39),('iws_bimanual_rope','Rope',277,39),('openh','Open-H',340,39)]
 for key,label,x,w in names:
  h=inv[key]['width_height'][1]/inv[key]['width_height'][0]*w
  c.rect(x-2,y+9,w+4,38,'#F3F6F8',r=3)
  photo(c,inv[key],x,y+11+(34-h)/2,w)
  c.text(x+w/2,y+55,label,8,MUTED,'center')

def evidence(c,rows,gain,y=190):
 c.text(7,y+4,'136 / 141 improve · 5 regress',8,INK)
 for i,r in enumerate(rows):
  x=8+(i%47)*3.42;yy=y+14+(i//47)*3.6
  if r['gain_percent']>0:c.circle(x,yy,.95,TEAL)
  else:
   c.line([(x-1,yy-1),(x+1,yy+1)],CORAL,.8);c.line([(x-1,yy+1),(x+1,yy-1)],CORAL,.8)
 c.text(204,y+4,f'{gain:.2f}%',15,TEAL,bold=True)
 c.text(269,y+3,'lower error vs AR',8,INK)
 c.text(204,y+20,'DROID development · h10',8,MUTED)

def ar_inset(c,x,y,w=148):
 c.text(x,y,'Autoregression',8.5,MUTED)
 # Own history/output and own feedback; never uses a ShiftWM output.
 token(c,x+3,y+20,23,MUTED,False)
 c.line([(x+31,y+31.5),(x+47,y+31.5)],MUTED,.8,True,semantic='AR rolling feature history enters its predictor')
 c.circle(x+65,y+31.5,14,'#F3F5F7',MUTED,.7);c.text(x+65,y+31.5,'Predict',8,MUTED,'center')
 c.line([(x+81,y+31.5),(x+104,y+31.5)],MUTED,.8,True,semantic='AR produces its own feature forecast')
 token(c,x+109,y+20,23,MUTED)
 c.line([(x+121,y+46),(x+121,y+56),(x+15,y+56),(x+15,y+45)],MUTED,.8,True,semantic='AR inserts its own forecast into the next three-state history')
 # The return arrow itself depicts rolling; the caption states history update.
 cmd(c,x+65,y+7,(x+65,y+13))

def fixed_path(c,x,y,w=181):
 c.text(x,y,'ShiftWM (ours)',9,TEAL)
 # One observed reference object and one schematic output, not repeated grids.
 token(c,x+2,y+21,30,BLUE,False)
 c.text(x+17,y+49,'Retain',8,BLUE,'center')
 c.line([(x+36,y+29),(x+68,y+29)],BLUE,1.2,True,semantic='Fixed original observed feature support conditions every queried horizon')
 c.rect(x+72,y+16,45,27,'#ECF8F5',TEAL,r=6,lw=.8);c.text(x+94.5,y+29,'Forecast',8,TEAL,'center')
 c.line([(x+121,y+29),(x+149,y+29)],TEAL,1.1,True,semantic='Fixed-source action-conditioned decoder emits a feature forecast; no output feedback')
 token(c,x+153,y+21,23,TEAL)
 cmd(c,x+95,y+2,(x+95,y+13))
 c.text(x+127,y+49,'Feature output',8,TEAL,'center')

def draft_a(inv,rows,gain):
 c=B.Canvas('camera-hero',396,216)
 c.text(9,9,'Forecast from what was observed',9,INK)
 camera(c,inv['droid'],15,28,136)
 # Photographic source and retained-feature source are linked, via encoder abstract.
 c.line([(164,71),(183,71),(183,95),(208,95)],BLUE,1,True,semantic='DROID observed video is encoded into the fixed observed-feature reference')
 fixed_path(c,209,66,181)
 ar_inset(c,216,6)
 # AR initialization alias is separate from the fixed decoded output.
 c.circle(205,34,2,BLUE);c.line([(208,34),(216,34)],BLUE,.8,True,semantic='Same initial observed information initializes the independent AR history')
 strip(c,inv,124);evidence(c,rows,gain,188)
 return c

def draft_b(inv,rows,gain):
 c=B.Canvas('observation-stream',396,216)
 # Wide input dominates; scope groups are left-to-right, not dataflow arrows.
 strip(c,inv,8)
 camera(c,inv['droid'],15,87,113)
 c.text(148,82,'Keep the source, query the future',8.5,INK)
 fixed_path(c,188,97,181)
 c.line([(139,119),(157,119),(157,126),(186,126)],BLUE,1,True,semantic='Recorded observations supply fixed encoded source features')
 c.text(148,159,'AR: forecast → next input',8,MUTED)
 token(c,286,158,22,MUTED);token(c,340,158,22,MUTED)
 c.line([(312,166),(335,166)],MUTED,.8,True,semantic='AR own forecast feeds the next step')
 c.line([(355,178),(355,185),(296,185),(296,177)],MUTED,.7,True,semantic='Independent AR recursion, not a ShiftWM feedback path')
 evidence(c,rows,gain,190);return c

def draft_c(inv,rows,gain):
 c=B.Canvas('camera-reference-hub',396,216)
 c.text(8,9,'Recorded input → fixed evidence → feature forecast',9,INK)
 camera(c,inv['droid'],133,29,127)
 c.text(8,36,'Autoregression',8.5,MUTED)
 token(c,21,57,24,MUTED);c.circle(85,65,14,'#F3F5F7',MUTED,.7);c.text(85,65,'F',9,MUTED,'center')
 c.line([(51,65),(69,65)],MUTED,.8,True)
 c.line([(100,65),(112,65),(112,95),(32,95),(32,77)],MUTED,.8,True,semantic='AR independent feature prediction cycles into its own input')
 c.text(8,113,'Rolling predictions',8,MUTED)
 c.text(278,36,'ShiftWM (ours)',8.5,TEAL)
 c.line([(274,76),(292,76)],BLUE,1,True,semantic='Encoded fixed observations are the decoder reference')
 c.rect(295,62,63,28,'#ECF8F5',TEAL,r=7,lw=.8);c.text(326.5,76,'Retain + query',8,TEAL,'center')
 c.line([(362,76),(377,76)],TEAL,.9,True);token(c,371,96,16,TEAL)
 cmd(c,326,49,(326,59));cmd(c,85,40,(85,49))
 strip(c,inv,124);evidence(c,rows,gain,188);return c

def camera_icon(c,x,y):
 # Original compact camera geometry: an illustration, not an experimental device.
 c.rect(x+2,y+4,36,25,'#D6E3EC',r=5,alpha=.55)
 c.rect(x,y+2,36,25,INK,'#496582',r=5,lw=.65)
 c.rect(x+4,y-1,12,5,'#426790',r=1.8)
 c.rect(x+25,y-1,7,3,GOLD,r=1.2)
 c.rect(x+2,y+6,7,17,'#31567C',r=2)
 for r,col in ((11.5,'#C0D9E8'),(9.5,'#153A58'),(7.4,BLUE),(4.6,'#126477'),(2.4,'#64B8CB')):c.circle(x+25,y+14,r,col)
 c.circle(x+22,y+11,1.5,'#D6F2F4')

def reference_tile(c,x,y,w=28):
 c.rect(x+2,y+2,w,w,'#CCDCE7',r=3)
 c.rect(x-2,y-2,w+4,w+4,'white',BLUE,r=3,lw=.8)
 for i in range(4):
  for j in range(4):c.rect(x+j*w/4,y+i*w/4,w/4-.65,w/4-.65,B.lighten(BLUE,.2+.5*((i+j*2)%6)/6),r=.7)
 # Retention is a semantic pin on the frame, not an experimental patch marker.
 c.circle(x+w+1,y-1,3,BLUE);c.circle(x+w+1,y-1,1.1,'white')

def selected(inv,rows,gain):
 c=B.Canvas('camera-observation-story',396,216)
 camera_icon(c,8,13)
 c.text(54,16,'DROID · observed video',8.5,BLUE)
 # Untouched registered median-case support frames0/5/10. Stair-step placement
 # exposes the stream without occluding any scientific image.
 for i,(frame,x,y,w) in enumerate(((0,7,59,43),(5,57,51,49),(10,113,32,77))):
  path=ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{frame}.png'
  h=w*9/16
  c.rect(x-2,y-2,w+4,h+4,'white','#ADC8DA',r=3,lw=.7)
  c.photo(path,x,y,w)
  if i<2:c.line([(x+w+2,y+h/2),(x+w+5,y+h/2)],BLUE,.7,True,semantic='Chronological order of recorded observed support frames, not model forecasts')
 c.text(9,99,'Encode and retain the observation',8,BLUE)
 # Observed initialization alias feeds only the independent AR comparator.
 c.text(212,18,'Same\nsupport',8,BLUE,'center');c.circle(217,34,2,BLUE);c.line([(220,34),(229,34)],BLUE,.8,True,semantic='Same initial observed information initializes the independent AR feature history')
 ar_inset(c,230,6)
 # Ours occupies the foreground and reads a fixed observed reference at every query.
 c.text(209,68,'ShiftWM (ours)',9,TEAL)
 c.line([(193,54),(200,54),(200,95),(207,95)],BLUE,1.05,True,semantic='The recorded DROID support is encoded; fixed last-observed features supply the retained reference')
 reference_tile(c,211,81,27)
 c.text(224,119,'Retain features',8,BLUE,'center')
 c.line([(243,95),(261,95)],BLUE,1.05,True,semantic='The retained last-observed feature source enters the learned gated mixer')
 c.rect(265,82,58,27,'#EAF7F3',TEAL,r=6,lw=.85)
 c.text(294,89,'Mix +',8.5,TEAL,'center');c.text(294,101,'correct',8.5,CORAL,'center')
 cmd(c,293,71,(293,79));c.text(303,71,'Actions',8,GOLD)
 # Both horizon queries are independently conditioned by observed support and
 # their own causal commands. Shared box is a schematic endpoint-query function.
 c.line([(326,95),(339,95),(346,85),(367,85)],TEAL,1,True,semantic='A horizon query produces a schematic future feature output, not RGB')
 c.line([(339,95),(339,115),(367,115)],TEAL,1,True,semantic='Another horizon query reuses the same observed source; no forecast feedback')
 token(c,371,75,20,TEAL);token(c,371,105,20,TEAL)
 c.text(381,65,'h=1',8,TEAL,'center');c.text(353,105,'h=10',8,TEAL,'center')
 strip(c,inv,128);evidence(c,rows,gain,192)
 return c

def checked_support():
 replay=ROOT/'paper/figure_sources/spatial_qualitative/replay.json'
 p=json.loads(replay.read_text());case=next(v for v in p['cases'] if v['prefix']=='case1')
 assert case['episode_id']=='droid-2dce8777c34ed372dc9ff50c'
 rec=[]
 for frame in (0,5,10):
  r=next(r for r in case['frame_exports'] if r['native_frame_index']==frame)
  path=replay.parent/r['path'];im=np.asarray(Image.open(path).convert('RGB'))
  assert B.sha(path)==r['file_sha256']
  assert hashlib.sha256(im.tobytes()).hexdigest()==r['decoded_pixel_sha256']
  rec.append({'path':str(path.relative_to(ROOT)),'file_sha256':B.sha(path),'pixel_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'frame':frame})
 return {'episode_id':case['episode_id'],'selection':'previously registered median episode-gain case, first eligible window, observed support0/5/10','replay_sha256':B.sha(replay),'assets':rec}

def native(c,path):
 receipt=c.drawio(path);tree=ET.parse(path)
 # Absolute arc size preserves the actual rounded exterior radii used by PDF.
 for e in c.elements:
  if e['kind']=='rect' and e['r']>0:
   cell=tree.find(f'.//mxCell[@id="{e["id"]}"]')
   cell.set('style',cell.get('style').replace('arcSize=8;',f'absoluteArcSize=1;arcSize={2*e["r"]};'))
 tree.write(path,encoding='utf-8',xml_declaration=True)
 receipt['rounding']='absolute native arc size = twice canonical corner radius'
 receipt['dense_activation_edges']=0
 return receipt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--drafts',action='store_true');args=parser.parse_args()
 DESIGN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'teaser-camera-story-v1','image.composite_image':False})
 p,m,rows,gain,a=B.checked_evidence();inv=inventory();support=checked_support();qa={}
 if args.drafts:
  for k,f in [('A',draft_a),('B',draft_b),('C',draft_c)]:
   c=f(inv,rows,gain);qa[k]=c.render(DESIGN/f'composition_{k}')
   (DESIGN/f'geometry_{k}.json').write_text(json.dumps({'elements':c.elements,'edges':c.edges,'photos':c.photos},indent=2)+'\n')
  thumbs=[Image.open(DESIGN/f'composition_{k}_paper_width.png').convert('RGB') for k in 'ABC'];sheet=Image.new('RGB',(thumbs[0].width*3,thumbs[0].height+25),'white');d=ImageDraw.Draw(sheet)
  for i,(k,im) in enumerate(zip('ABC',thumbs)):sheet.paste(im,(i*im.width,25));d.text((i*im.width+8,5),k,fill='black')
  sheet.save(DESIGN/'contact_sheet.png')
  (DESIGN/'draft_evidence.json').write_text(json.dumps({'arithmetic':a,'qa':qa,'selection':'A, refined with actual three-observation stream and compact original vector camera'},indent=2)+'\n')
 c=selected(inv,rows,gain)
 assert len(c.photos)==11
 expected={r['asset_sha256'] for r in inv.values()}|{r['file_sha256'] for r in support['assets']}
 assert {r['sha256'] for r in c.photos}==expected
 basename=OUT/'teaser_camera_story';n=native(c,basename.with_suffix('.drawio'));qa=c.render(basename)
 (PUBLIC/'geometry.json').write_text(json.dumps({'elements':c.elements,'edges':c.edges,'photos':c.photos,'size_points':[396,216]},indent=2)+'\n')
 (OUT/'teaser_camera_story_caption.tex').write_text(CAPTION+'\n')
 (OUT/'teaser_camera_story_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_camera_story.pdf}\n\\caption[Forecast from retained observations.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
 refs=[PUBLIC/'reference/composition_reference.png',PUBLIC/'reference/generation_manifest.json',PUBLIC/'web_reference/source.json',PUBLIC/'web_reference/camera.svg',PUBLIC/'web_reference/LICENSE.txt']
 deps=[Path(__file__),ROOT/'paper/scripts/render_teaser_benchmarks.py',ROOT/'src/shiftwm/real_video_spatial/model.py',B.SRC/'population.json',B.SRC/'asset_manifest.json',ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json',ROOT/'paper/figure_sources/benchmark_gallery/asset_manifest.json',ROOT/'paper/figure_sources/spatial_qualitative/replay.json']+refs
 evidence={'status':'candidate_pending_independent_review','source_sha256':{str(q.relative_to(ROOT)):B.sha(q) for q in deps},'authoritative_source_sha256':p['authoritative_source_sha256'],'size_inches':[5.5,3.0],'arithmetic':a,'episode_marks':[{'episode_id':r['episode_id'],'gain_percent':r['gain_percent'],'mark':'circle' if r['gain_percent']>0 else 'cross'} for r in rows],'support':support,'photos':c.photos,'semantic_edges':c.edges,'semantics':{'recorded_images':'eleven unchanged full RGB images represent nine data sources; DROID uses three registered observed support frames','fixed_interface':'all three encoded observed support states plus last-observed source Z0; collapsed backbone/context inference omitted','causality':'each symbolic endpoint query uses its own supplied causal action prefix','ar':'rolling three-state feature history; original inferred context fixed; own output feedback only','forecast_glyphs':'illustrative features, not RGB, physical positions or measured vector values; two schematic queries, ten-step measured horizon','gallery':'historical context models distinct; IWS training inputs only; Open-H physical phantom input audit only','camera':'original editable vector illustration, not a photographed instrument'},'qa':qa,'native':n,'caption':CAPTION,'references':{'generated':'actually inspected composition reference; no synthetic data imagery embedded','web':'actually inspected official Lucide camera silhouette and full license retained; no SVG paths copied'},'skills':['paper-visual-design','paper-figure-creation','codex-paper-figure-skill'],'outputs_sha256':{ext:B.sha(basename.with_suffix('.'+ext)) for ext in ('pdf','svg','png','drawio')}}
 (OUT/'teaser_camera_story_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 (PUBLIC/'asset_manifest.json').write_text(json.dumps({'support':support,'images':c.photos,'scope_inventory':'paper/figure_sources/benchmark_gallery/benchmark_inventory.json','source_sha256':evidence['source_sha256'],'no_crop_or_recolor':True,'generated_images_embedded':False},indent=2)+'\n')
 print(json.dumps({'qa':qa,'source':B.sha(Path(__file__)),'outputs':evidence['outputs_sha256']},indent=2))
if __name__=='__main__':main()
