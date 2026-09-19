#!/usr/bin/env python3
"""Structural, scientific-edge and conservative geometry checks; not a GUI render."""
from pathlib import Path
import hashlib,json,re,shutil,xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import ImageFont
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED_INPUTS={
'spatial-encoder':{'observed-history'},'encoded-support':{'spatial-encoder'},'spatial-mean':{'encoded-support'},
'support-context':{'spatial-mean','past-junction'},'film':{'encoded-support','support-context'},
'past-junction':{'past-actions'},'action-concat':{'past-junction','future-actions'},'prefix-gru':{'action-concat'},
'action-selector':{'prefix-gru'},'temporal-predictor':{'film','action-selector'},'horizon-state':{'temporal-predictor'},
'gate':{'hidden-for-gate'},'one-minus-gate':{'gate'},'identity-product':{'anchor','one-minus-gate'},
'query-projection':{'hidden-for-query'},'key-projection':{'encoding-for-key'},'row-softmax':{'query-projection','key-projection'},
'normalized-forecast':{'innovation-merge'},'innovation-merge':{'anchor-blend','bounded-innovation'},
'anchor-blend':{'identity-product','gated-mix-product'},'gated-mix-product':{'mixed-anchor','gate'},
'mixed-anchor':{'transport-product'},'transport-product':{'anchor','transport-matrix'},'transport-matrix':{'row-softmax'},
'residual-projection':{'hidden-for-innovation'},'bounded-innovation':{'residual-projection'}}
EXPECTED_ALIASES={'anchor':'observed-history','hidden-for-query':'horizon-state','hidden-for-gate':'horizon-state','hidden-for-innovation':'horizon-state','encoding-for-key':'encoded-support'}

def main():
 path=OUT/'spatial-transport-architecture.drawio'; tree=ET.parse(path); graph=tree.find('./diagram/mxGraphModel');root=graph.find('root');cells=root.findall('mxCell'); ids=[c.attrib['id']for c in cells]
 assert len(ids)==len(set(ids)), 'Duplicate cell IDs'
 byid={c.attrib['id']:c for c in cells};assert byid['0'].attrib=={'id':'0'} and byid['1'].attrib=={'id':'1','parent':'0'}
 assert graph.attrib['adaptiveColors']=='auto' and not any('image='in c.attrib.get('style','')for c in cells),'Raster embedding forbidden'
 ledger=json.loads((OUT/'canonical-graph.json').read_text()); nodes=ledger['nodes'];edges=ledger['edges'];actual=[];issues=[]
 assert '<!--' not in path.read_text(), 'XML comments forbidden by skill'
 assert len(edges)==sum(len(v)for v in EXPECTED_INPUTS.values()), 'Duplicate or missing semantic connector'
 byedge={e['id']:e for e in edges}
 for c in cells[2:]:
  assert c.attrib['parent']in byid
  geo=c.find('mxGeometry');assert geo is not None
  if c.attrib.get('edge')=='1':
   assert geo.attrib.get('relative')=='1' and geo.attrib.get('as')=='geometry'
   assert c.attrib['source']in nodes and c.attrib['target']in nodes
   expected=byedge[c.attrib['id']]
   style=dict(item.split('=',1)for item in c.attrib['style'].split(';')if '='in item)
   assert style['endArrow']=='block' and style['endFill']=='1' and 'startArrow'not in style,'Arrow direction changed'
   assert [float(style[k])for k in ['exitX','exitY']]==expected['exit'] and [float(style[k])for k in ['entryX','entryY']]==expected['entry'],'Connector ports differ from checked ledger'
   actual_points=[[float(p.attrib['x']),float(p.attrib['y'])]for p in geo.findall('./Array/mxPoint')]
   assert actual_points==expected['waypoints'],'XML waypoint geometry differs from checked ledger'
   actual.append((c.attrib['id'],c.attrib['source'],c.attrib['target']))
  else:
   assert c.attrib.get('vertex')=='1' and c.attrib['id']in nodes
   assert c.attrib.get('value')==nodes[c.attrib['id']]['label'] and c.attrib['parent']==nodes[c.attrib['id']]['parent'],'Native editable label/parent differs'
   assert [float(geo.attrib[k])for k in ['x','y','width','height']]==nodes[c.attrib['id']]['bbox']
 assert set(actual)=={(e['id'],e['source'],e['target'])for e in edges},'XML/semantic edge disagreement'
 inputs=defaultdict(set)
 for e in edges:inputs[e['target']].add(e['source'])
 assert dict(inputs)==EXPECTED_INPUTS, 'Required model inputs missing, invented or reversed'
 assert {k:v['source']for k,v in ledger['aliases'].items()}==EXPECTED_ALIASES,'Cross-panel aliases changed'
 for k,v in ledger['sources'].items():assert sha(ROOT/k)==v,'Scientific source changed'
 assert ledger['equations']=={'transport':'T_h = softmax_row(Q_h K_0^T / sqrt(96) + 4 I)','gate':'g_h = sigmoid(W_g H_h + b_g)','innovation':'Delta_h = tanh(R_h)','forecast':'Zhat_h = (1-g_h)*Z_0 + g_h*(T_h @ Z_0) + Delta_h'}
 def absolute(id):
  x,y,w,h=nodes[id]['bbox'];parent=nodes[id]['parent']
  if parent!='1':
   px,py,_,_=absolute(parent);x+=px;y+=py
  return x,y,w,h
 def port(id,uv):
  x,y,w,h=absolute(id);return(x+uv[0]*w,y+uv[1]*h)
 for id,n in nodes.items():
  x,y,w,h=absolute(id)
  if min(w,h)<=0 or x<0 or y<0 or x+w>1520 or y+h>1540:issues.append({'kind':'canvas_bounds','node':id})
  if n['parent']!='1':
   px,py,pw,ph=absolute(n['parent'])
   if x<px-.01 or y<py-.01 or x+w>px+pw+.01 or y+h>py+ph+.01:issues.append({'kind':'group_bounds','node':id})
  if n['label']:
   font=int(re.search('fontSize=(\d+)',byid[id].attrib['style']).group(1));f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',font)
   widths=[f.getlength(line)for line in n['label'].split('\n')];height=len(widths)*font*1.15
   pad=0 if n['kind']in('heading','label','operator')else 12
   if max(widths)>w-pad+1 or height>h-pad+1:issues.append({'kind':'estimated_text_fit','node':id,'max_width':max(widths),'text_height':height,'box':[w,h]})
 paths={}
 for e in edges:
  points=[port(e['source'],e['exit'])]+[tuple(p)for p in e['waypoints']]+[port(e['target'],e['entry'])];paths[e['id']]=points
  for a,b in zip(points,points[1:]):
   if abs(a[0]-b[0])>.01 and abs(a[1]-b[1])>.01:issues.append({'kind':'nonorthogonal_edge','edge':e['id'],'segment':[a,b]})
   if a==b:continue
   for nid,n in nodes.items():
    if nid in (e['source'],e['target']) or nid.startswith('panel-') or n['parent']!='1':continue
    x,y,w,h=absolute(nid);x+=2;y+=2;w-=4;h-=4
    if abs(a[0]-b[0])<.01:
     hit=x<a[0]<x+w and max(min(a[1],b[1]),y)<min(max(a[1],b[1]),y+h)
    else:hit=y<a[1]<y+h and max(min(a[0],b[0]),x)<min(max(a[0],b[0]),x+w)
    if hit:issues.append({'kind':'connector_crosses_unrelated_node','edge':e['id'],'node':nid,'segment':[a,b]})
 for a_idx,e in enumerate(edges):
  for other in edges[a_idx+1:]:
   if {e['source'],e['target']}&{other['source'],other['target']}:continue
   for a,b in zip(paths[e['id']],paths[e['id']][1:]):
    for c,d in zip(paths[other['id']],paths[other['id']][1:]):
     # Crossing check for perpendicular interiors; endpoints may meet only at declared common nodes.
     if a[0]==b[0] and c[1]==d[1] and min(c[0],d[0])<a[0]<max(c[0],d[0]) and min(a[1],b[1])<c[1]<max(a[1],b[1]):
      issues.append({'kind':'unrelated_connector_crossing','edges':[e['id'],other['id']]})
     if a[1]==b[1] and c[0]==d[0] and min(a[0],b[0])<c[0]<max(a[0],b[0]) and min(c[1],d[1])<a[1]<max(c[1],d[1]):
      issues.append({'kind':'unrelated_connector_crossing','edges':[e['id'],other['id']]})
 result={'status':'structural_semantic_and_conservative_geometry_passed'if not issues else'geometry_revision_required','xml_sha256':sha(path),'canonical_graph_sha256':sha(OUT/'canonical-graph.json'),'generator_sha256':sha(OUT/'build_drawio.py'),'validator_sha256':sha(__file__),'cells':len(cells),'vertices':sum(c.attrib.get('vertex')=='1'for c in cells),'connectors':len(edges),'native_patch_and_matrix_cells':sum(n['kind']=='cell'for n in nodes.values()),'root_cells_verified':True,'unique_cell_ids':True,'all_connectors_have_relative_geometry':True,'canonical_required_inputs_verified':True,'canonical_aliases_verified':True,'no_embedded_raster_objects':True,'issues':issues,'drawio_cli':shutil.which('drawio')or shutil.which('draw.io'),'actual_drawio_export':'skipped: draw.io CLI not found; existing reviewed Figure24 PDF remains unchanged','actual_drawio_visual_review':'not performed; no draw.io export exists. Checks here are XML/source/geometry checks, not a claimed application screenshot.','text_measurement_scope':'Conservative DejaVu Sans measurements approximate native Helvetica labels; actual app font/layout remains unverified.'}
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if issues:raise SystemExit(1)

if __name__=='__main__':main()
