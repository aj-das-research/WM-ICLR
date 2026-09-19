#!/usr/bin/env python3
"""Compact recorded-action forecasting task using the verified original example."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.path import Path as MP
from matplotlib.colors import Normalize
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/editorial_task';PACK=ROOT/'paper/figure_sources/spatial_qualitative'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED_REPLAY='12d805f0692f9cd0068f7eb392cf9be996dd778f1fff00f5c1ee4273cfffac20'
EXPECTED_SHARED_STD='8c3cdb672eb46d548ab9b80c4701cc2b27937dde3d3f5d6b94406eafd2286de8'
EXPECTED_ARRAYS='816c6e918835c89df64b28f4dabee991546aab6cc7c03741269c2d17daf17cb3'
CAPTION=r'''\textbf{Forecasting from recorded observations and supplied actions.} Three observed frames, two past action blocks and ten supplied future blocks determine the future-feature forecast. The withheld frame enters only the evaluator through the same frozen encoder. Each block spans five source-frame intervals; indices denote stored frames, not seconds. Images are unchanged DROID observations (CC BY 4.0); feature glyphs are schematic. The heatmap shows three-seed standardized feature error for the prespecified median episode's first window (MSE 0.1727; gain $-0.305\%$ versus autoregression). The model generates neither RGB nor robot actions. Full selection and comparisons appear in Appendix~\ref{app:spatial-development}.'''


def checked_inputs():
    assert sha(PACK/'replay.json')==EXPECTED_REPLAY
    j=json.loads((PACK/'replay.json').read_text());assert sha(PACK/'replay_arrays.npz')==j['arrays_sha256']==EXPECTED_ARRAYS
    with np.load(PACK/'replay_arrays.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    case=j['cases'][1];prefix=case['prefix'];assert prefix=='case1' and case['rank_descending']==70 and case['first_window_start']==0
    assert a[prefix+'_actions'].shape==(12,35) and a[prefix+'_features'].shape==(13,6144)
    assert np.array_equal(a[prefix+'_frame_indices'],np.arange(0,61,5))
    images=[];image_sources={}
    for export,slot in zip(case['frame_exports'],(0,1,2,12)):
        p=PACK/export['path'];pixels=np.asarray(Image.open(p))
        assert sha(p)==export['file_sha256'] and hashlib.sha256(pixels.tobytes()).hexdigest()==export['decoded_pixel_sha256']
        assert np.array_equal(pixels,a[prefix+'_images'][slot])
        images.append(pixels);image_sources[str(p.relative_to(ROOT))]=sha(p)
    assert sha(DESIGN/'shared_channel_std.json')==EXPECTED_SHARED_STD
    norm=json.loads((DESIGN/'shared_channel_std.json').read_text());assert norm['source_sha256']==j['sources'][norm['source_path']]
    # Independent NumPy arithmetic from raw saved predictions and targets.
    std=np.repeat(np.array(norm['channel_std'],np.float32),16);assert std.shape==(6144,) and np.isfinite(std).all() and np.all(std>0)
    reconstructed=[];max_difference=0.;numerical_checks=[]
    for c in j['cases']:
        key=c['prefix'];target=a[key+'_features'][12]
        for mode in ('autoregressive','transport'):
            seed_maps=[]
            for seed in (0,1,2):
                predicted=a[f'{key}_{mode}_s{seed}_predictions'][9]
                independently_computed=np.square((predicted-target)/std).reshape(384,4,4).mean(0,dtype=np.float64)
                recorded=a[f'{key}_{mode}_s{seed}_patch_errors'][9]
                diff=float(np.max(np.abs(independently_computed-recorded)));max_difference=max(max_difference,diff)
                assert np.allclose(independently_computed,recorded,rtol=2e-6,atol=2e-7)
                assert np.isclose(independently_computed.mean(),a[f'{key}_{mode}_s{seed}_native_mse'][9],rtol=2e-6,atol=2e-7)
                numerical_checks.append({'case':key,'mode':mode,'seed':seed,'maximum_patch_reconstruction_difference':diff})
                seed_maps.append(recorded)
            reconstructed.append(np.mean(seed_maps,axis=0))
    vmax=max(float(x.max()) for x in reconstructed)
    error=np.mean([a[f'{prefix}_transport_s{s}_patch_errors'][9] for s in (0,1,2)],axis=0)
    mean=float(error.mean());assert np.isclose(mean,case['first_window_endpoint_mean']['transport'],rtol=2e-6,atol=2e-7)
    baseline=case['first_window_endpoint_mean']['autoregressive'];gain=100*(baseline-case['first_window_endpoint_mean']['transport'])/baseline
    assert np.isclose(gain,case['first_window_gain_percent'],rtol=1e-12)
    assert f'{mean:.4f}'=='0.1727' and f'{gain:.3f}'=='-0.305'
    return images,error,vmax,{'replay_sha256':EXPECTED_REPLAY,'arrays_sha256':EXPECTED_ARRAYS,'image_sha256':image_sources,
        'shared_scale_source_sha256':sha(DESIGN/'shared_channel_std.json'),'case':case['episode_id'],'rank':70,'first_window':0,
        'observed_source_indices':[0,5,10],'target_source_index':60,'past_action_blocks':2,'supplied_future_action_blocks':10,
        'source_intervals_per_block':5,'native_action_coordinates_per_row':7,'grouped_action_width':35,
        'error_mean':mean,'gain_vs_autoregression_percent':gain,'common_error_range':[0,vmax],
        'patch_error':error.tolist(),'independent_reconstruction_checks':numerical_checks,'maximum_patch_reconstruction_difference':max_difference}


def render():
    images,error,vmax,evidence=checked_inputs();OUT.mkdir(parents=True,exist_ok=True)
    style=json.loads((ROOT/'paper/design/editorial_style.json').read_text());C=style['colors'];ink=C['ink'];blue=C['observation'];orange=C['action'];green=C['ours'];gray=C['secondary']
    plt.rcParams.update({'font.family':style['figure_font_family'],'font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    W,H=396,201.6;fig=plt.figure(figsize=(5.5,2.8),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off');texts=[];edges=[];nodes={}
    def text(x,y,s,size=8.5,color=ink,weight='normal',ha='center'):
        t=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,zorder=10);texts.append(t)
    def box(name,x,y,w,h,label,color,fill='white'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',fc=fill,ec=color,lw=.75,zorder=3));text(x+w/2,y+h/2,label,size=8.5,color=color);nodes[name]=[x,y,w,h]
    def arrow(name,points,color=gray):
        ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=6.5,color=color,lw=.9,zorder=5));edges.append({'id':name,'points':points,'color':color})
    def actual_image(x,y,w,pixels,name):
        h=w*pixels.shape[0]/pixels.shape[1];ia=fig.add_axes([x/W,y/H,w/W,h/H]);ia.imshow(pixels,interpolation='nearest');ia.set_axis_off();nodes[name]=[x,y,w,h]
    def features(x,y,s,color,name):
        for row in range(4):
            for col in range(4):
                xx=x+col*s/4;yy=y+row*s/4
                ax.add_patch(Rectangle((xx,yy),s/4,s/4,fc='white',ec=color,lw=.5,zorder=5))
                for line in range(3):ax.plot([xx+1.2,xx+s/4-1.2],[yy+1.5+line*1.5]*2,color=color,lw=.45,zorder=6)
        nodes[name]=[x,y,s,s]
    # Forecasting lane: no image from the future appears here.
    text(72,161,'Observed images',weight='bold',color=blue)
    for i,idx in enumerate((0,5,10)):
        actual_image(9+i*43,126,40,images[i],f'observed-{idx}');text(29+i*43,118,str(idx),size=8,color=gray)
    text(72,107,'Source frame index',size=8,color=gray)
    box('forecast-encoder',157,122,45,34,'Frozen\nencoder',blue,'#F2F7FA')
    arrow('observed-to-encoder',[(136,139),(156,139)],blue)
    box('predictor',232,122,71,34,'World model',green,'#F3F8F3')
    arrow('encoded-history-to-predictor',[(202,139),(231,139)],blue)
    text(270,194,'Recorded commands',size=9.5,weight='bold',color=orange)
    text(228,183,'2 past',size=8,color=orange);text(294,183,'10 supplied future',size=8,color=orange)
    for i in range(2):ax.add_patch(Rectangle((220+i*8,168),6,8,fc='#ECD1B7',ec=orange,lw=.45,zorder=4))
    for i in range(10):ax.add_patch(Rectangle((251+i*5.5,168),4,8,fc='#F5E6D8',ec=orange,lw=.45,zorder=4))
    ax.plot([220,220,304.5,304.5],[165,163,163,165],color=orange,lw=.65,zorder=4)
    arrow('past-and-future-actions-to-predictor',[(268,163),(268,157)],orange)
    features(337,125,28,green,'predicted-features');text(350,161,'Future features',size=8.5,weight='bold',color=green)
    arrow('predictor-to-future-features',[(303,139),(336,139)],green)
    # Evaluator-only data boundary, with no upward data path to the model.
    ax.add_patch(FancyBboxPatch((5,6),386,88,boxstyle='round,pad=0,rounding_size=4',fc='#F6F8FA',ec='#BEC9D2',lw=.75,zorder=0))
    text(13,86,'EVALUATION ONLY',size=8.5,weight='bold',color=gray,ha='left')
    text(58,74,'Withheld frame 60',size=8.5)
    actual_image(19,29,72,images[3],'withheld-rgb')
    box('target-encoder',119,37,58,33,'Same frozen\nencoder',blue)
    arrow('withheld-to-target-encoder',[(92,53.5),(118,53.5)],blue)
    features(215,40,27,blue,'target-features');text(228.5,75,'Target features',size=8.5,color=blue)
    arrow('target-encoder-to-target-features',[(177,53.5),(214,53.5)],blue)
    ax.add_patch(Circle((284,53.5),10,fc='white',ec=gray,lw=.8,zorder=4));text(284,53.5,'MSE',size=8);nodes['comparison']=[274,43.5,20,20]
    arrow('target-features-to-comparison',[(242,53.5),(273,53.5)],blue)
    arrow('prediction-to-comparison',[(351,124),(351,103),(284,103),(284,64)],green)
    ia=fig.add_axes([325/W,40/H,28/W,28/H]);ia.imshow(error,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');ia.set_xticks([]);ia.set_yticks([])
    for spine in ia.spines.values():spine.set_color(gray);spine.set_linewidth(.5)
    text(340,75,'Patch error',size=8.5,weight='bold');text(337,28,'Lower is better',size=8,color=gray)
    arrow('comparison-to-error-map',[(295,53.5),(324,53.5)],gray)
    cbax=fig.add_axes([362/W,40/H,4/W,28/H]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,vmax),cmap='magma'),cax=cbax,ticks=[0,vmax]);cb.ax.set_yticklabels(['0',f'{vmax:.3f}']);cb.ax.tick_params(labelsize=8,length=2,pad=2);cb.outline.set_linewidth(.4)
    text(58,17,'Recorded target RGB',size=8,color=gray);text(231,17,'Visual-feature comparison',size=8,color=gray)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[t.get_window_extent(renderer) for t in texts]
    overlaps=[]
    for i,a in enumerate(bounds):
        for j,b in enumerate(bounds[i+1:],i+1):
            if a.overlaps(b):overlaps.append([texts[i].get_text(),texts[j].get_text()])
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.x1>fig.bbox.width or b.y0<0 or b.y1>fig.bbox.height]
    if overlaps or clipped:raise ValueError(json.dumps({'overlaps':overlaps,'clipped':clipped}))
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'task_main.{ext}',dpi=300)
    fig.savefig(OUT/'task_paper_size.png',dpi=110);fig.savefig(OUT/'task_enlarged.png',dpi=360);plt.close(fig)
    ImageOps.grayscale(Image.open(OUT/'task_paper_size.png')).save(OUT/'task_grayscale.png')
    evidence.update(status='candidate_pending_actual_pixel_review',renderer_sha256=sha(__file__),style_sha256=sha(ROOT/'paper/design/editorial_style.json'),
        geometry={'width_inches':5.5,'height_inches':2.8,'minimum_font_pt':8,'text_overlaps':overlaps,'clipped':clipped},arrows=edges,nodes=nodes,
        image_attribution='DROID CC BY4.0, https://droid-dataset.github.io/ and https://creativecommons.org/licenses/by/4.0/',
        limitations='The actual patch-error map is a validation diagnostic, not pixel reconstruction or a demonstrated successful policy. Feature glyphs are symbolic vectors.')
    (OUT/'task_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');(OUT/'task_caption.tex').write_text(CAPTION+'\n')
    (OUT/'task_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/task_main.pdf}\n\\caption[Recorded-action forecasting task.]{'+CAPTION+'}\n\\label{fig:editorial-task}\n\\end{figure}\n')
    print(json.dumps({'shape_inches':[5.5,2.8],'mse':evidence['error_mean'],'gain_percent':evidence['gain_vs_autoregression_percent'],'max_error_reconstruction_difference':evidence['maximum_patch_reconstruction_difference'],'layout_issues':[]}))

if __name__=='__main__':render()
