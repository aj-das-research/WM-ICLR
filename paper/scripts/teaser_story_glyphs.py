#!/usr/bin/env python3
"""Editable schematic latent-feature glyphs; never RGB patches or measured values.

Coordinates are abstract points on an equal-aspect Matplotlib axis. Functions do
not change axes, fonts, or global style. All returned ports are (x, y) tuples.
Rear planes suggest channel depth; four-by-four front positions suggest space.
Three visible planes are an illustration, NOT a claim of three model channels.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle
from matplotlib.path import Path as MPath

INK = '#26364B'
COBALT = '#315EDB'
TEAL = '#129B99'
CORAL = '#E96E64'
GOLD = '#E8AE3D'
PALETTE = (COBALT, TEAL, CORAL, GOLD)


def _mix(color, white):
    return tuple((1-white)*v+white for v in to_rgb(color))


def _check(w, h):
    if not np.isfinite([w, h]).all() or w <= 0 or h <= 0:
        raise ValueError('Glyph bounds must be finite and positive')


def _ports(x, y, w, h):
    return {'left': (x,y+h/2), 'right': (x+w,y+h/2),
            'top': (x+w/2,y+h), 'bottom': (x+w/2,y),
            'center': (x+w/2,y+h/2)}


def feature_tensor(ax, x, y, w, h, role):
    """Draw a 4×4 spatial face with subtle channel layering.

    role: 'observed'/'anchor', 'forecast'/'mixed', or 'recursive'. Aliases
    'input', 'fixed', 'prediction', and 'corrected' are accepted. Roles alter
    the accent only; glyph values never encode measured accuracy or drift.
    Recommend 36×30pt minimum; 42×34pt offers more clear channel depth.
    Returns {'bounds', 'ports', 'artists', 'semantics'}.
    """
    _check(w,h)
    aliases={'input':'observed','fixed':'anchor','prediction':'forecast','corrected':'mixed'}
    role=aliases.get(role,role)
    if role not in ('observed','anchor','forecast','mixed','recursive'):
        raise ValueError('Unknown feature-tensor role: '+str(role))
    accent=COBALT if role in ('observed','anchor') else TEAL if role in ('forecast','mixed') else CORAL
    artists=[]
    def add(p):ax.add_patch(p);artists.append(p);return p
    depth=min(w*.095,h*.105);fw=w-2*depth;fh=h-2*depth
    radius=min(fw,fh)*.045
    # Three shifted feature planes show channel depth without illegible stripes.
    for i,color in ((2,GOLD),(1,accent),(0,accent)):
        add(FancyBboxPatch((x+i*depth,y+i*depth),fw,fh,
            boxstyle=f'round,pad=0,rounding_size={radius}',
            facecolor=_mix(color,.80 if i else .95),edgecolor=_mix(color,.05),
            linewidth=.62,zorder=2+(2-i)*.1))
    # A translucent side face visually joins planes into one structured tensor.
    add(Polygon([(x+fw,y),(x+fw+2*depth,y+2*depth),(x+fw+2*depth,y+fh+2*depth),(x+fw,y+fh)],
                closed=True,facecolor=_mix(accent,.80),edgecolor='none',zorder=2.15))
    inset=min(fw,fh)*.06;gap=min(fw,fh)*.022
    cw=(fw-2*inset-3*gap)/4;ch=(fh-2*inset-3*gap)/4
    pattern=np.array([[0,0,1,1],[0,2,1,3],[2,2,3,3],[0,1,3,2]])
    for r in range(4):
        for c in range(4):
            px=x+inset+c*(cw+gap);py=y+inset+(3-r)*(ch+gap)
            color=PALETTE[pattern[r,c]]
            # Stable patch identities across roles; same objects, no invented motion.
            fill=_mix(color,.06+.08*((r+c)%3))
            add(FancyBboxPatch((px,py),cw,ch,boxstyle=f'round,pad=0,rounding_size={min(cw,ch)*.18}',
                              facecolor=fill,edgecolor='none',zorder=3))
    # One retained source identity; high-contrast frame survives grayscale.
    px=x+inset+(cw+gap);py=y+inset+2*(ch+gap)
    add(FancyBboxPatch((px-.23,py-.23),cw+.46,ch+.46,
                      boxstyle=f'round,pad=0,rounding_size={min(cw,ch)*.20}',
                      facecolor='none',edgecolor=INK,linewidth=.7,zorder=4))
    return {'bounds':(x,y,w,h),'ports':_ports(x,y,w,h),'artists':artists,
            'semantics':'Schematic 4×4 spatial positions with compressed channel depth; colors/planes are illustrative, not actual latent values.'}


def mixing_vignette(ax, x, y, w, h):
    """Three latent pieces → one mixture, plus a distinct bounded correction.

    Recommended 110×48pt (minimum100×44). Text-free except 8pt + and Σ
    mathematical operators. The little amber marker is held inside two end
    stops: a schematic bounded feature correction, not an action or motion.
    Returns bounds/ports/artists plus named internal ports and arrow ledger.
    """
    _check(w,h);artists=[];arrows=[]
    def xy(u,v):return (x+u*w,y+v*h)
    def add(p):ax.add_patch(p);artists.append(p);return p
    def arrow(name,points,color):
        pts=[xy(*p) for p in points]
        patch=FancyArrowPatch(path=MPath(pts,[MPath.MOVETO]+[MPath.LINETO]*(len(pts)-1)),
            arrowstyle='-|>',mutation_scale=4.6,linewidth=.75,color=color,zorder=3)
        add(patch);arrows.append({'id':name,'points':pts})
    def chip(u,v,color,scale=1):
        cx,cy=xy(u,v);cw=.135*w*scale;ch=.19*h*scale;d=min(cw,ch)*.12
        add(Polygon([(cx-cw/2,cy+ch/2),(cx-cw/2+d,cy+ch/2+d),(cx+cw/2+d,cy+ch/2+d),(cx+cw/2,cy+ch/2)],
                    closed=True,facecolor=_mix(color,.48),edgecolor='none',zorder=4))
        add(Polygon([(cx+cw/2,cy-ch/2),(cx+cw/2+d,cy-ch/2+d),(cx+cw/2+d,cy+ch/2+d),(cx+cw/2,cy+ch/2)],
                    closed=True,facecolor=_mix(color,.15),edgecolor='none',zorder=4))
        add(FancyBboxPatch((cx-cw/2,cy-ch/2),cw,ch,boxstyle=f'round,pad=0,rounding_size={min(cw,ch)*.13}',
                          facecolor=color,edgecolor='white',linewidth=.4,zorder=5))
        # Small inset facets read as a vector-valued piece, not an RGB crop.
        for j,amount in enumerate((.10,.35,.60)):
            add(Circle((cx+(-.26+j*.26)*cw,cy),min(cw,ch)*.105,facecolor=_mix(color,amount),edgecolor='none',zorder=6))
        return {'left':(cx-cw/2,cy),'right':(cx+cw/2+d,cy),'center':(cx,cy)}
    source_ports=[]
    for i,(v,color) in enumerate(((.81,COBALT),(.52,TEAL),(.23,CORAL))):
        source_ports.append(chip(.10,v,color))
        # Distinct entry ports converge into one sum; there are no crossings.
        arrow('source'+str(i)+'-to-mixture',[(.187,v),(.355,.52+(v-.52)*.28)],color)
    radius=min(w*.043,h*.09);cx,cy=xy(.40,.52)
    add(Circle((cx,cy),radius,facecolor='white',edgecolor=INK,linewidth=.7,zorder=5))
    txt=ax.text(cx,cy,r'$\Sigma$',ha='center',va='center',fontsize=8,color=INK,zorder=6);artists.append(txt)
    # Schematic convex color blend; not three regions concatenated into a patch.
    mixed=tuple(np.average([to_rgb(c) for c in (COBALT,TEAL,CORAL)],axis=0,weights=[.45,.35,.20]))
    arrow('mixture-to-latent-piece',[(.45,.52),(.527,.52)],TEAL)
    output=chip(.61,.52,mixed,1.05)
    addx,addy=xy(.805,.52);r=min(w*.038,h*.08)
    add(Circle((addx,addy),r,facecolor='white',edgecolor=GOLD,linewidth=.7,zorder=5))
    artists.append(ax.text(addx,addy,'+',ha='center',va='center',fontsize=8,color=INK,zorder=6))
    arrow('mixture-to-correction-add',[(.695,.52),(.763,.52)],TEAL)
    # A terminal feature object makes the result explicit rather than ending in space.
    finished=chip(.935,.52,TEAL,.68)
    arrow('corrected-feature-output',[(.849,.52),(.935-.135*.68/2,.52)],TEAL)
    # Two explicit stops surround the correction chip; no decorative tanh plot.
    left,right=xy(.68,.115),xy(.94,.115)
    line=ax.plot([left[0],right[0]],[left[1],right[1]],color=GOLD,lw=.7,zorder=2)[0];artists.append(line)
    for u in (.68,.94):
        a,b=xy(u,.055),xy(u,.175);artists.append(ax.plot([a[0],b[0]],[a[1],b[1]],color=GOLD,lw=.8,zorder=3)[0])
    mx,my=xy(.805,.115);side=min(w*.052,h*.105)
    add(FancyBboxPatch((mx-side/2,my-side/2),side,side,boxstyle=f'round,pad=0,rounding_size={side*.16}',facecolor=GOLD,edgecolor='white',lw=.4,zorder=4))
    arrow('bounded-correction-to-add',[(.805,.185),(.805,.422)],GOLD)
    return {'bounds':(x,y,w,h),'ports':_ports(x,y,w,h),'source_ports':source_ports,
            'mixture_port':output,'finished_output_ports':finished,'output_port':finished['right'],'correction_port':xy(.805,.115),
            'artists':artists,'arrows':arrows,
            'semantics':'Three schematic latent vectors mix; a distinct range-limited feature innovation is added. No RGB movement, optical flow, actual weights, or measured result.'}


def preview(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(5.5,1.8),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,396),ylim=(0,129.6));ax.axis('off')
    calls=[]
    for xx,role in ((14,'anchor'),(78,'forecast'),(142,'recursive')):
        calls.append(feature_tensor(ax,xx,61,42,34,role));ax.text(xx+21,48,role,ha='center',fontsize=8,color=INK)
    calls.append(mixing_vignette(ax,217,48,158,64));ax.text(296,31,'Latent mixture + bounded correction',ha='center',fontsize=8,color=INK)
    calls.append(feature_tensor(ax,14,11,36,30,'anchor'));ax.text(64,26,'36 × 30 pt minimum tensor',va='center',fontsize=8,color=INK)
    ax.text(14,116,'Original vector glyphs · schematic latent features',fontsize=8.5,color=INK)
    ax.text(217,12,'Preview labels are external to the API.',fontsize=8,color=INK)
    fig.canvas.draw()
    for ext in ('pdf','svg','png'):fig.savefig(output/('preview.'+ext),dpi=300)
    fig.savefig(output/'paper_width.png',dpi=110);plt.close(fig)
    ImageOps.grayscale(Image.open(output/'paper_width.png')).save(output/'grayscale.png')
    receipt={'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dimensions_inches':[5.5,1.8],'tensor_recommendation_points':[42,34],'tensor_minimum_points':[36,30],'mix_recommendation_points':[110,48],'mix_minimum_points':[100,44],'roles':['observed','anchor','forecast','mixed','recursive'],'rendering':'Original vector geometry only. Preview labels are outside the functions; operator text is8pt.','semantics':[c['semantics'] for c in calls],'mix_arrow_ledger':calls[3]['arrows'],'status':'candidate_pending_actual_pixel_review'}
    (output/'preview_evidence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    fig=plt.figure(figsize=(5.5,1.45),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,396),ylim=(0,104.4));ax.axis('off')
    feature_tensor(ax,12,43,36,30,'anchor');feature_tensor(ax,63,39,42,34,'forecast')
    mixing_vignette(ax,126,36,110,44);mixing_vignette(ax,276,36,100,44)
    for xx,label in ((30,'36 × 30'),(84,'42 × 34'),(181,'110 × 44'),(326,'100 × 44')):
        ax.text(xx,23,label+' pt',ha='center',fontsize=8,color=INK)
    ax.text(12,91,'Exact physical sizes · 8 pt mathematical operators',fontsize=8.5,color=INK)
    ax.text(12,7,'All colors, feature pieces and bounds are schematic.',fontsize=8,color=INK)
    for ext in ('pdf','svg','png'):fig.savefig(output/('minimum_sizes.'+ext),dpi=300)
    fig.savefig(output/'minimum_sizes_paper_width.png',dpi=110);plt.close(fig)
    ImageOps.grayscale(Image.open(output/'minimum_sizes_paper_width.png')).save(output/'minimum_sizes_grayscale.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preview',default='paper/design/teaser_story/glyphs');args=p.parse_args();preview(args.preview)
