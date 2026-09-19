#!/usr/bin/env python3
"""Recorded-video input/evaluator contract, using the frozen verified example."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MPath
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/editorial"
PROOF = OUT / "task_refined_proof"
HELPER = ROOT / "paper/scripts/render_editorial_task.py"
HELPER_SHA = "2d93b0d64ee9954f73cef476c83eb4a7189db75e3006f4f79325898ed03ed495"
INCLUDE = ROOT / "paper/generated/real_video/spatial_task_figure.tex"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SLATE, BLUE, TEAL, AMBER = "#394B59", "#4477AA", "#287C7A", "#B88746"
GRAY = "#677682"
W, H = 396.0, 194.4
CAPTION = (
    r"\textbf{Available inputs and withheld evaluation targets.} Three observed frames "
    r"(0, 5, 10), two past action blocks and ten supplied future blocks determine the "
    r"feature forecast. Frame 60 enters only the evaluator through the same frozen "
    r"DINOv2 encoder. One block spans five source-frame intervals, not seconds. "
    r"Tiles are schematic feature grids; the heatmap is the actual three-seed "
    r"standardized $h=10$ patch error for the prespecified median-by-episode-gain "
    r"case's first window (MSE 0.1727; gain $-0.305\%$ versus autoregression). "
    r"Its linear scale retains the original 0--0.787 range. Recorded DROID images "
    r"are unchanged (CC BY 4.0); no RGB or robot actions are generated."
)


def inputs():
    if sha(HELPER) != HELPER_SHA:
        raise ValueError("The reviewed input verifier changed")
    spec = importlib.util.spec_from_file_location("refined_task_verified_source", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checked_inputs()


def theme():
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8.2,
                         "mathtext.fontset": "stix", "svg.fonttype": "none",
                         "svg.hashsalt": "shiftwm-task-refined-v1", "pdf.fonttype": 42,
                         "image.composite_image": False})


def sketches():
    """Three structurally different task grammars, not cosmetic variants."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    plans = [
        ("A  Separate forecast / evaluator lanes",
         [(0.12,.77,"observed"),(.43,.77,"encoder"),(.71,.77,"predict"),(.94,.77,"forecast"),
          (.12,.23,"withheld"),(.43,.23,"same encoder"),(.71,.23,"target"),(.94,.23,"error")],
         [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7),(3,7)]),
        ("B  Central comparator",
         [(0.14,.77,"observed + actions"),(.54,.77,"predict"),(.85,.50,"MSE"),
          (.14,.23,"withheld RGB"),(.54,.23,"encode target")],
         [(0,1),(1,2),(3,4),(4,2)]),
        ("C  Timeline and private scoring inset",
         [(0.12,.76,"past"),(.39,.76,"last real frame"),(.80,.76,"withheld frame"),
          (.34,.32,"predict prefix"),(.74,.32,"score features")],
         [(0,1),(1,3),(3,4),(2,4)])]
    for ax, (title, nodes, edges) in zip(axes, plans):
        ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
        ax.text(.01,1.02,title,fontsize=9,color=SLATE)
        for i,j in edges:
            x,y,_=nodes[i];xx,yy,_=nodes[j]
            ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle="-|>",
                         shrinkA=23,shrinkB=23,mutation_scale=7,lw=.7,color=GRAY))
        for x,y,label in nodes:
            ax.add_patch(FancyBboxPatch((x-.11,y-.085),.22,.17,
                         boxstyle="round,pad=.005",fc="#F4F7F8",ec=BLUE,lw=.6))
            ax.text(x,y,label,ha="center",va="center",fontsize=8,color=SLATE)
        if title.startswith("A"):
            ax.plot([0,1],[.5,.5],color=GRAY,lw=.6,ls="--")
    fig.tight_layout()
    fig.savefig(OUT / "task_refined_sketches.pdf", metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(OUT / "task_refined_sketches.png", dpi=150)
    plt.close(fig)


def render():
    images, error, vmax, verified = inputs()
    OUT.mkdir(parents=True,exist_ok=True);PROOF.mkdir(exist_ok=True)
    theme(); sketches()
    brief = {
        "mode":"Existing recorded-video forecasting task adapted to the registered spatial evaluation",
        "source":"paper/sections/main_evaluation.tex and the frozen replay verifier",
        "one_sentence":"Predict from available images and recorded commands, then score against features of a withheld future frame without giving that frame to the model.",
        "slot":"Attached appendix, 5.5 by 2.7 inches, Liberation Sans 8.0–8.3 pt; headings 8.7 pt",
        "selected_composition":"A: paired input/evaluator lanes; the single downward prediction branch makes the access boundary explicit.",
        "rejected_compositions":{"B":"Central merge gives the two encoder routes less room and encourages ambiguous crossings.","C":"Timeline is intuitive but makes encoder identity and the target-to-score route too small."},
        "asset_policy":"Only original recorded images and actual median error; simple authored patch tiles are schematic. No generated media or invented measurements.",
        "scope":"Task and information access, not another decoder architecture or claimed new benchmark."
    }
    (OUT / "task_refined_brief.json").write_text(json.dumps(brief,indent=2)+"\n")
    fig=plt.figure(figsize=(5.5,2.7),facecolor="white")
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis("off")
    texts=[];edges=[];nodes={}
    def text(x,y,label,size=8.2,color=SLATE,ha="center"):
        t=ax.text(x,y,label,ha=ha,va="center",fontsize=size,color=color,zorder=9)
        texts.append(t);return t
    def block(name,x,y,w,h,label,color=SLATE,frozen=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=2",
                     fc="white",ec=color,lw=.7,ls="--" if frozen else "-",zorder=3))
        text(x+w/2,y+h/2,label,color=color);nodes[name]=[x,y,w,h]
    def arrow(name,points,color=GRAY):
        ax.add_patch(FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),
                     arrowstyle="-|>",mutation_scale=6,color=color,lw=.8,zorder=4))
        edges.append({"id":name,"points":points,"color":color})
    def photo(name,x,y,w,pixels):
        h=w*pixels.shape[0]/pixels.shape[1]
        ia=fig.add_axes([x/W,y/H,w/W,h/H]);ia.imshow(pixels,interpolation="none");ia.axis("off")
        nodes[name]=[x,y,w,h]
    def tiles(name,x,y,s,color):
        # Sixteen semantic patch slots, no decorative tiny-line glyphs.
        levels=np.array([[.16,.35,.21,.48],[.44,.71,.31,.24],[.29,.47,.62,.36],[.56,.22,.38,.50]])
        from matplotlib.colors import to_rgb
        rgb=np.array(to_rgb(color))
        for row in range(4):
            for col in range(4):
                fill=1-levels[row,col]*(1-rgb)
                ax.add_patch(Rectangle((x+col*s/4,y+(3-row)*s/4),s/4,s/4,
                             fc=fill,ec="white",lw=.45,zorder=5))
        ax.add_patch(Rectangle((x,y),s,s,fc="none",ec=color,lw=.65,zorder=6))
        nodes[name]=[x,y,s,s]

    text(70,178,"Observed frames",size=8.7,color=BLUE)
    for i,idx in enumerate((0,5,10)):
        x=8+i*42;photo(f"observed-{idx}",x,135,39,images[i])
        text(x+19.5,127,str(idx),size=8,color=GRAY)
    block("forecast-encoder",145,134,51,26,"Frozen\nDINOv2",BLUE,True)
    arrow("observed-only-to-frozen-encoder",[(132,146.8),(144,146.8)],BLUE)
    block("predictor",217,134,69,26,"Predictor",TEAL)
    arrow("encoded-history-to-predictor",[(197,147),(216,147)],BLUE)
    text(259,187,"Actions: 2 past + 10 supplied future",size=8,color=AMBER)
    for i in range(2):ax.add_patch(Rectangle((225+7*i,176),5,6,fc=AMBER,ec=AMBER,lw=.4))
    for i in range(10):ax.add_patch(Rectangle((249+4.6*i,176),3.3,6,fc="#F1E7D8",ec=AMBER,lw=.4))
    ax.plot([225,225,293.7,293.7],[173,171,171,173],color=AMBER,lw=.6)
    arrow("exact-two-past-ten-query-blocks-to-predictor",[(259,171),(259,161)],AMBER)
    text(349,178,"Forecast features",size=8.7,color=TEAL)
    text(349,166,r"$\widehat Z_{10}$",size=8,color=TEAL)
    tiles("predicted-feature-grid",336,134,26,TEAL)
    arrow("predictor-to-feature-forecast",[(287,147),(335,147)],TEAL)

    ax.add_patch(FancyBboxPatch((5,6),386,98,boxstyle="round,pad=0,rounding_size=3",
                 fc="#F3F5F7",ec="#CAD2D8",lw=.6,zorder=0))
    text(14,96,"Evaluation only",size=8.7,color=SLATE,ha="left")
    text(53,81,"Withheld frame 60",size=8.2)
    photo("withheld-only-for-evaluation",12,29,82,images[3])
    text(53,18,"Recorded target",size=8,color=GRAY)
    block("target-encoder",121,37,55,30,"Same frozen\nencoder",BLUE,True)
    arrow("withheld-frame-to-same-frozen-encoder",[(95,52),(120,52)],BLUE)
    tiles("target-feature-grid",211,38,28,BLUE)
    text(225,81,"Target features",size=8.2,color=BLUE)
    text(225,26,r"$Z_{10}$",size=8,color=BLUE)
    arrow("target-encoder-to-target-features",[(177,52),(210,52)],BLUE)
    ax.add_patch(Circle((284,52),12,fc="white",ec=SLATE,lw=.75,zorder=4))
    text(284,52,"MSE",size=8);nodes["mse"]=[272,40,24,24]
    arrow("target-features-to-mse",[(240,52),(271,52)],BLUE)
    arrow("prediction-down-to-evaluation-only",[(349,133),(349,114),(284,114),(284,65)],TEAL)
    error_ax=fig.add_axes([326/W,38/H,28/W,28/H])
    error_ax.imshow(error,cmap="magma",vmin=0,vmax=vmax,interpolation="nearest")
    error_ax.set_xticks([]);error_ax.set_yticks([])
    for spine in error_ax.spines.values():spine.set_color(GRAY);spine.set_linewidth(.5)
    text(340,81,"Patch MSE",size=8.2)
    text(340,24,"Linear scale",size=8,color=GRAY)
    arrow("mse-to-measured-patch-error",[(297,52),(325,52)],SLATE)
    cbax=fig.add_axes([362/W,38/H,4/W,28/H])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,vmax),cmap="magma"),cax=cbax,ticks=[0,vmax])
    cb.ax.set_yticklabels(["0",f"{vmax:.3f}"]);cb.ax.tick_params(labelsize=8,length=2,pad=2);cb.outline.set_linewidth(.4)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=[t.get_window_extent(renderer) for t in texts]
    overlap=[(texts[i].get_text(),texts[j].get_text()) for i,a in enumerate(bounds)
             for j,b in enumerate(bounds) if j>i and a.overlaps(b)]
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
    if overlap or clipped:raise ValueError(json.dumps({"overlap":overlap,"clipped":clipped}))
    for ext in ("pdf","svg","png"):
        metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else ({"Date":None} if ext=="svg" else None)
        fig.savefig(OUT/f"task_refined.{ext}",dpi=300,metadata=metadata)
    plt.close(fig)
    for name,flags,dpi in (("paper_width",[],120),("grayscale",["-gray"],120),("detail",[],300)):
        subprocess.run(["pdftoppm","-singlefile","-png","-r",str(dpi),*flags,str(OUT/"task_refined.pdf"),str(PROOF/name)],check=True,capture_output=True)
    previous=OUT/"task_refined_previous_include.tex"
    if not previous.exists():previous.write_bytes(INCLUDE.read_bytes())
    (OUT/"task_refined_caption.tex").write_text(CAPTION+"\n")
    INCLUDE.write_text("\\begin{figure}[!htb]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{generated/editorial/task_refined.pdf}\n"
        "\\caption[Recorded inputs and evaluator-only target.]{"+CAPTION+"}\n"
        "\\label{fig:spatial-task}\n\\end{figure}\n")
    tex=("\\documentclass{article}\n\\usepackage{iclr2027_conference,times,graphicx,amsmath}\n"
         "\\begin{document}\n\\input{generated/real_video/spatial_task_figure}\n\\end{document}\n")
    (PROOF/"proof.tex").write_text(tex)
    env=dict(os.environ,TEXINPUTS="template/official/iclr2027:"+os.environ.get("TEXINPUTS",""))
    for i in (1,2):
        run=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",
            "-output-directory=generated/editorial/task_refined_proof","generated/editorial/task_refined_proof/proof.tex"],cwd=ROOT/"paper",env=env,capture_output=True,text=True)
        (PROOF/f"compile_{i}.txt").write_text(run.stdout+run.stderr)
        if run.returncode:raise ValueError("Official task proof failed")
    log=(PROOF/"proof.log").read_text()
    warnings=[line for line in log.splitlines() if any(w in line for w in ("Overfull","Underfull","Float too large","LaTeX Warning","undefined","! LaTeX"))]
    if warnings:raise ValueError("Task proof warnings: "+repr(warnings))
    subprocess.run(["pdftoppm","-singlefile","-png","-r","120",str(PROOF/"proof.pdf"),str(PROOF/"official_page")],check=True,capture_output=True)
    outputs=[OUT/f"task_refined.{e}" for e in ("pdf","svg","png")]+[INCLUDE,OUT/"task_refined_caption.tex",PROOF/"proof.pdf"]
    evidence={"status":"source_numeric_and_geometry_passed_visual_review_pending","created_utc":datetime.now(timezone.utc).isoformat(),
        "renderer_sha256":sha(__file__),"helper_sha256":sha(HELPER),"verified_inputs":verified,
        "geometry":{"inches":[5.5,2.7],"body_font_pt":8.2,"minimum_font_pt":8,"maximum_heading_pt":8.7,"text_overlaps":overlap,"clipped":clipped},
        "font":"Liberation Sans regular; STIX math","palette":{"slate":SLATE,"blue":BLUE,"teal":TEAL,"amber":AMBER},
        "nodes":nodes,"edges":edges,"source_error_coordinates":"native shared-channel-standardized 4x4 patch errors; actual three-seed h10 median-case first window",
        "information_access":"No withheld frame route into predictor; the forecast crosses into the evaluation lane only. Both frozen encoders use the same identity.",
        "feature_glyphs":"16 filled patch tiles; schematic, with no tiny-line glyphs or generated RGB.","error_scale":"Original linear 0 to actual maximum across all six replay maps; not the logarithmic main-gallery mapping.",
        "proof_warnings":warnings,"outputs":{str(p.relative_to(ROOT)):sha(p) for p in outputs}}
    (OUT/"task_refined_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps({"error_mean":verified["error_mean"],"gain_percent":verified["gain_vs_autoregression_percent"],"shape":[5.5,2.7],"overlap":overlap,"clipped":clipped,"proof_warnings":warnings}))


if __name__ == "__main__":
    render()
