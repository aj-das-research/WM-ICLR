"""Small shared point-coordinate scene for PDF/SVG/PNG and editable draw.io.

No figure content, experiment imports, or old-figure runtime dependencies.
Coordinates are points, with the origin at the upper left. Native application
export/review is separate: paper/scripts/export_drawio_browser.py SOURCE PREFIX.
"""
from pathlib import Path
import base64
import hashlib
import html
import json
import math
import xml.etree.ElementTree as ET
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MPath
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Scene:
    def __init__(self, width=396, height=230, root=None):
        self.width, self.height = float(width), float(height)
        self.root = Path(root or Path.cwd()).resolve()
        self.items = []

    def add(self, kind, **kw):
        item = {"id": kw.pop("id", f"{kind}-{len(self.items)+1:04}"),
                "kind": kind, **kw}
        if item["id"] in {"0", "1"} or any(p["id"] == item["id"] for p in self.items):
            raise ValueError("Scene IDs must be unique and cannot be 0 or 1")
        self.items.append(item)
        return item["id"]

    def rect(self, x, y, w, h, fill="white", stroke="#B8C8D8", r=2,
             lw=.65, dash=False, **kw):
        return self.add("rect", x=x, y=y, w=w, h=h, fill=fill,
                        stroke=stroke, r=r, lw=lw, dash=dash, **kw)

    def circle(self, x, y, r=5, fill="white", stroke="#287C7A", lw=.8, **kw):
        return self.add("circle", x=x, y=y, r=r, fill=fill, stroke=stroke, lw=lw, **kw)

    def poly(self, points, fill, stroke="none", lw=.6, **kw):
        return self.add("polygon", points=points, fill=fill, stroke=stroke, lw=lw, **kw)

    def text(self, x, y, text, size=8, color="#20354C", align="left", bold=False, **kw):
        if align not in {"left", "center", "right"}:
            raise ValueError("Unsupported horizontal alignment")
        return self.add("text", x=x, y=y, text=text, size=size, color=color,
                        align=align, bold=bold, **kw)

    def edge(self, points, color="#687C90", head=True, dash=False, lw=.8, **kw):
        if len(points) < 2:
            raise ValueError("An edge needs at least two points")
        return self.add("edge", points=points, color=color, head=head, dash=dash, lw=lw, **kw)

    def line(self, points, color="#687C90", dash=False, lw=.8, **kw):
        return self.edge(points, color=color, head=False, dash=dash, lw=lw, **kw)

    def photo(self, x, y, w, path, h=None, **kw):
        """Place the full source array; h optionally specifies an aspect-fit box."""
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        relative = path.relative_to(self.root)
        with Image.open(path) as im:
            iw, ih = im.size
            if im.format != "PNG":
                raise ValueError("Use a source PNG so the exact file embeds without conversion")
        fw, fh = float(w), float(w) * ih / iw
        if h is not None:
            scale = min(float(w) / iw, float(h) / ih)
            fw, fh = iw * scale, ih * scale
            x, y = x + (w-fw)/2, y + (h-fh)/2
        return self.add("image", x=x, y=y, w=fw, h=fh, path=str(relative),
                        sha256=sha(path), source_size_px=[iw, ih], **kw)

    def _photo_path(self, p):
        path = self.root / p["path"]
        if sha(path) != p["sha256"]:
            raise ValueError(f"Source image changed: {path}")
        return path

    def _figure(self):
        plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8,
                             "mathtext.fontset": "stix", "pdf.fonttype": 42,
                             "svg.fonttype": "none", "svg.hashsalt": "droid-setup-v1",
                             "image.composite_image": False})
        fig = plt.figure(figsize=(self.width/72, self.height/72), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, self.width), ylim=(self.height, 0))
        ax.axis("off")
        texts = {}
        for index, p in enumerate(self.items):
            k, z = p["kind"], index+1
            if k == "rect":
                ax.add_patch(FancyBboxPatch((p["x"], p["y"]), p["w"], p["h"],
                    boxstyle=f"round,pad=0,rounding_size={p['r']}", facecolor=p["fill"],
                    edgecolor=p["stroke"], lw=p["lw"], ls="--" if p["dash"] else "-", zorder=z))
            elif k == "circle":
                ax.add_patch(Circle((p["x"], p["y"]), p["r"], facecolor=p["fill"],
                                   edgecolor=p["stroke"], lw=p["lw"], zorder=z))
            elif k == "polygon":
                ax.add_patch(Polygon(p["points"], facecolor=p["fill"],
                                     edgecolor=p["stroke"], lw=p["lw"], zorder=z))
            elif k == "text":
                texts[p["id"]] = ax.text(p["x"], p["y"], p["text"], ha=p["align"], va="center",
                    fontsize=p["size"], color=p["color"], weight="bold" if p["bold"] else "normal", zorder=z)
            elif k == "edge":
                path = MPath(p["points"], [MPath.MOVETO]+[MPath.LINETO]*(len(p["points"])-1))
                ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if p["head"] else "-",
                    mutation_scale=4.8, lw=p["lw"], color=p["color"],
                    ls=(0, (2, 1.8)) if p["dash"] else "-", zorder=z))
            elif k == "image":
                ax.imshow(np.asarray(Image.open(self._photo_path(p))),
                    extent=(p["x"], p["x"]+p["w"], p["y"]+p["h"], p["y"]),
                    interpolation="none", aspect="auto", zorder=z)
            else:
                raise ValueError(f"Unsupported scene kind: {k}")
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        bounds = {}
        for ident, t in texts.items():
            b = t.get_window_extent(ren)
            bounds[ident] = {"bbox_pt": [b.x0*72/fig.dpi, self.height-b.y1*72/fig.dpi,
                b.x1*72/fig.dpi, self.height-b.y0*72/fig.dpi], "font_pt": t.get_fontsize()}
        return fig, bounds

    def render(self, base):
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig, bounds = self._figure()
        overlaps, clipped = [], []
        for ident, record in bounds.items():
            a = record["bbox_pt"]
            if a[0] < 0 or a[1] < 0 or a[2] > self.width or a[3] > self.height:
                clipped.append(ident)
            for other, rr in bounds.items():
                b = rr["bbox_pt"]
                if ident < other and min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    overlaps.append([ident, other])
        for ext in ("pdf", "svg", "png"):
            metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else ({"Date": None} if ext == "svg" else {"Software": "ShiftWM shared scene"})
            fig.savefig(base.with_suffix("."+ext), dpi=300, metadata=metadata)
        plt.close(fig)
        qa = {"text_overlap_ids": overlaps, "clipped_text_ids": clipped,
              "minimum_font_pt": min((v["font_pt"] for v in bounds.values()), default=None)}
        geometry = {"canvas_pt": [self.width, self.height], "objects": self.items,
                    "text_bounds": bounds, "layout_checks": qa}
        base.with_suffix(".geometry.json").write_text(json.dumps(geometry, indent=2)+"\n")
        return qa

    def native(self, path):
        """Export the same objects; this does not claim native application review."""
        fig, measured = self._figure()
        plt.close(fig)
        doc = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
        diagram = ET.SubElement(doc, "diagram", {"id": "droid-setup", "name": "DROID setup"})
        model = ET.SubElement(diagram, "mxGraphModel", {"page": "1", "pageScale": "1",
            "pageWidth": str(self.width), "pageHeight": str(self.height), "math": "0", "shadow": "0"})
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        for p in self.items:
            k = p["kind"]
            at = {"id": p["id"], "parent": "1"}
            if k == "edge":
                at.update(edge="1", style=f"edgeStyle=none;rounded=0;strokeColor={p['color']};strokeWidth={p['lw']};endArrow={'block' if p['head'] else 'none'};endSize=3;startArrow=none;dashed={int(p['dash'])};")
                cell = ET.SubElement(root, "mxCell", at)
                geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
                pts = p["points"]
                for point, role in [(pts[0], "sourcePoint"), (pts[-1], "targetPoint")]:
                    ET.SubElement(geo, "mxPoint", {"x": str(point[0]), "y": str(point[1]), "as": role})
                arr = ET.SubElement(geo, "Array", {"as": "points"})
                for x, y in pts[1:-1]:
                    ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})
                continue
            at["vertex"] = "1"
            x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("w", 1), p.get("h", 1)
            if k == "text":
                b = measured[p["id"]]["bbox_pt"]
                w, h = b[2]-b[0]+1.4, b[3]-b[1]+.8
                x -= {"left": 0, "center": w/2, "right": w}[p["align"]]
                y -= h/2
                at["value"] = html.escape(p["text"]).replace("\n", "<br>")
                at["style"] = f"text;html=1;align={p['align']};verticalAlign=middle;whiteSpace=nowrap;overflow=visible;spacing=0;fontFamily=Liberation Sans;fontSize={p['size']};fontColor={p['color']};fontStyle={int(p['bold'])};strokeColor=none;fillColor=none;"
            elif k == "rect":
                at["style"] = f"rounded={int(p['r']>0)};absoluteArcSize=1;arcSize={2*p['r']};fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};dashed={int(p['dash'])};"
            elif k == "circle":
                x -= p["r"]; y -= p["r"]; w = h = 2*p["r"]
                at["style"] = f"ellipse;fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};"
            elif k == "image":
                data = base64.b64encode(self._photo_path(p).read_bytes()).decode()
                at["style"] = f"shape=image;imageAspect=0;aspect=fixed;image=data:image/png,{data};"
            elif k == "polygon":
                pts = p["points"]; x = min(q[0] for q in pts); y = min(q[1] for q in pts)
                w = max(q[0] for q in pts)-x; h = max(q[1] for q in pts)-y
                if min(w, h) <= 0:
                    raise ValueError("Use line() for a zero-area polygon")
                shape = ET.Element("shape", {"name": "editable-polygon", "w": str(w), "h": str(h), "aspect": "variable", "strokewidth": "inherit"})
                fg = ET.SubElement(shape, "foreground"); pn = ET.SubElement(fg, "path")
                for j, (xx, yy) in enumerate(pts):
                    ET.SubElement(pn, "move" if j == 0 else "line", {"x": str(xx-x), "y": str(yy-y)})
                ET.SubElement(pn, "close"); ET.SubElement(fg, "fillstroke")
                compressor = zlib.compressobj(wbits=-15)
                encoded = base64.b64encode(compressor.compress(ET.tostring(shape))+compressor.flush()).decode()
                at["style"] = f"shape=stencil({encoded});fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};"
            else:
                raise ValueError(f"Unsupported scene kind: {k}")
            cell = ET.SubElement(root, "mxCell", at)
            ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(doc); ET.ElementTree(doc).write(path, encoding="utf-8", xml_declaration=True)
        return {"objects": len(self.items), "source_sha256": sha(path),
                "native_application_render": "Not performed by this serializer"}

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'paper/figure_sources/droid_forecast_setup'
OUT=ROOT/'paper/generated/editorial'
INK='#233951';BLUE='#297CC1';TEAL='#138D88';GOLD='#BE8424';VIOLET='#8065AB';GRAY='#748496';LINE='#D3DFE8'
ASSETS=PUBLIC/'assets'
def asset(cam,frame):
 return ASSETS/f'{cam}_f{frame:03d}.png'
def panel(s,x,y,w,h,letter,title,col=BLUE):
 s.rect(x,y,w,h,'#FFFFFF',LINE,r=4,lw=.55)
 s.text(x+5,y+10,letter,size=9,color=col,bold=True)
 s.text(x+19,y+10,title,size=8.5,color=INK,bold=True)
def cam(s,x,y,col=BLUE,scale=1):
 s.rect(x,y,14*scale,9*scale,'#EEF4F8',col,r=1.6,lw=.7)
 s.rect(x+3*scale,y-2*scale,5*scale,2*scale,col,col,r=.6,lw=.3)
 s.circle(x+7*scale,y+4.5*scale,3*scale,'white',col,.65)
 s.circle(x+7*scale,y+4.5*scale,1.4*scale,col,col,.3)
def rig(s,x,y):
 # Conceptual camera roles only; positions are not calibrated extrinsics.
 s.poly([(x+15,y+38),(x+40,y+23),(x+72,y+38),(x+46,y+54)],'#E7E0D3','#B5AC9E',.5)
 for a,b in [((20,41),(20,55)),((46,52),(46,65)),((67,40),(67,53))]:s.edge([(x+a[0],y+a[1]),(x+b[0],y+b[1])],GRAY,False,lw=1.5)
 s.circle(x+29,y+36,6,'#CEDAE3','#90A1B0',.5)
 s.edge([(x+29,y+33),(x+35,y+15),(x+51,y+21),(x+53,y+35)],'#8AA1B4',False,lw=6)
 for a,b in [(29,33),(35,15),(51,21)]:s.circle(x+a,y+b,3.4,'#EAF0F5','#687D8E',.8)
 s.edge([(x+49,y+35),(x+49,y+40)],INK,False,lw=1.5);s.edge([(x+57,y+35),(x+57,y+40)],INK,False,lw=1.5)
 cam(s,x+3,y+15,BLUE,.8);cam(s,x+73,y+17,VIOLET,.8);cam(s,x+57,y+24,GRAY,.5)
 s.edge([(x+13,y+27),(x+33,y+41)],BLUE,False,True,.6);s.edge([(x+77,y+29),(x+58,y+42)],VIOLET,False,True,.6)
 s.text(x+7,y+9,'1',8,BLUE);s.text(x+80,y+11,'2',8,VIOLET)
 s.text(x+60,y+14,'W',8,GRAY)
def grid(s,x,y,w=20,col=TEAL):
 s.rect(x-1,y-1,w+2,w+2,'white',col,r=1.5,lw=.6)
 for j,a in enumerate([.22,.65,.45,.85]):
  import matplotlib.colors as mc
  rgb=mc.to_rgb(col);c=mc.to_hex(tuple(a*z+(1-a) for z in rgb))
  s.rect(x+(j%2)*w/2,y+(j//2)*w/2,w/2-.5,w/2-.5,c,'white',r=.2,lw=.2)
def encoder(s,x,y,w=47,h=28,label="Frozen\nDINOv2"):
 s.rect(x,y,w,h,'#F1F6FB',BLUE,r=3,lw=.65,dash=True)
 s.text(x+w/2,y+h/2,label,8,BLUE,'center')
def actions(s,x,y,step=18):
 for j in range(7):
  col='#F5EAD7' if j<2 else '#E4C07C';s.rect(x+j*step,y,13,10,col,GOLD,r=1,lw=.4)
  for i in range(3):s.edge([(x+2+j*step,y+2+i*2.5),(x+10+j*step,y+2+i*2.5)],GOLD,False,lw=.35)
def choice_a():
 s=Scene(width=396,height=232,root=ROOT)
 panel(s,0,0,112,232,'A','Camera setup')
 rig(s,9,20);s.text(56,95,'Schematic',8,GRAY,'center')
 for camid,y,col,title,role in [('cam1',113,BLUE,'Camera 1','Train / val / test'),('cam2',161,VIOLET,'Camera 2','Transfer test')]:
  s.photo(6,y,45,asset(camid,10));s.text(57,y+6,title,8,col);s.text(57,y+20,role.replace('Train / val / test','Train + val\nPrimary test'),8,INK)
 s.photo(6,204,45,asset('wrist',10));s.text(57,210,'Wrist',8,GRAY);s.text(57,223,'Inspect only',8,GRAY)
 panel(s,120,0,276,134,'B','Predict future visual features',BLUE)
 for i,f in enumerate([0,5,10]):
  x=129+i*50;s.text(x+21,27,f'Frame {f}',8,BLUE,'center');s.photo(x,35,42,asset('cam1',f))
 s.edge([(129,63),(129,67),(271,67),(271,63)],BLUE,False,lw=.55)
 s.text(200,76,'Observed history',8,BLUE,'center')
 s.edge([(273,47),(287,47)],BLUE,semantic='Observed RGB history enters the frozen encoder only')
 encoder(s,289,32,49,29)
 s.edge([(314,62),(314,94)],BLUE,semantic='Past encoded visual features condition the predictor')
 s.text(197,91,'Recorded commands',8,GOLD,'center');actions(s,134,100)
 s.text(197,123,'2 past + 5 future blocks',8,GOLD,'center')
 s.edge([(258,105),(287,105)],GOLD,semantic='Two past and five supplied future command blocks condition forecasting')
 s.rect(289,94,50,24,'#EAF7F5',TEAL,r=4);s.text(314,106,'Forecast',8,TEAL,'center')
 s.edge([(340,106),(351,106)],TEAL,semantic='Recursive model outputs five future feature vectors')
 grid(s,354,95,19,TEAL);s.text(363,81,'Predicted\nfeatures',8,TEAL,'center');s.text(363,126,'Frame 35',8,TEAL,'center')
 panel(s,120,143,276,89,'C','Score the withheld future',GOLD)
 s.rect(124,160,202,65,'#FFF9EF','none',r=3,lw=0)
 s.text(162,170,'Frame 35',8,GOLD,'center');s.photo(132,179,61,asset('cam1',35))
 s.edge([(196,196),(210,196)],GOLD,semantic='Withheld future RGB goes only to evaluator encoding')
 encoder(s,212,182,55,29,label="Same frozen\nDINOv2")
 s.edge([(268,196),(277,196)],GOLD,semantic='Same frozen encoder and train-fitted normalization define fixed targets')
 grid(s,280,185,19,GOLD);s.text(289,218,'Target',8,GOLD,'center')
 s.edge([(301,196),(332,196)],GOLD,semantic='Ground-truth future features enter the comparison only')
 s.rect(334,184,52,28,'#F4F6F9',GRAY,r=3,lw=.6);s.text(360,198,'Feature\nerror',8,INK,'center')
 s.edge([(375,105),(390,105),(390,170),(360,170),(360,184)],TEAL,semantic='Predicted feature at the same horizon is compared with withheld target')
 s.text(360,222,'MSE / cosine',8,GRAY,'center')
 return s


def main():
 manifest=json.loads((PUBLIC/'manifest.json').read_text())
 for rel,digest in manifest['runtime_inputs'].items():
  if sha(ROOT/rel)!=digest:raise ValueError('Changed source input: '+rel)
 data=json.loads((PUBLIC/'data.json').read_text())
 OUT.mkdir(parents=True,exist_ok=True)
 s=choice_a();base=OUT/'droid_forecast_setup';review=s.render(base)
 assert not review['text_overlap_ids'] and not review['clipped_text_ids'],review
 assert review['minimum_font_pt']>=8
 native=s.native(PUBLIC/'droid_forecast_setup.drawio')
 im=Image.open(base.with_suffix('.png'));im.resize((660,round(660*s.height/s.width)),Image.Resampling.LANCZOS).save(OUT/'droid_forecast_setup_paper_width.png')
 im.convert('L').save(OUT/'droid_forecast_setup_grayscale.png')
 record={'status':'rendered_and_geometry_checked','canonical_source':'paper/scripts/render_droid_forecast_setup.py','source_sha256':sha(__file__),'width_inches':s.width/72,'height_inches':s.height/72,'geometry_review':review,'native':native,'source_manifest_sha256':sha(PUBLIC/'manifest.json'),'source_data_sha256':sha(PUBLIC/'data.json'),'scope':'Existing recorded DROID historical-study setup, not experimental outputs or current spatial architecture','photo_boundary':'Unchanged recorded raster observations; original editable vector camera/robot/feature glyphs are explicitly schematic','exports':{str(base.with_suffix(ext).relative_to(ROOT)):sha(base.with_suffix(ext)) for ext in ['.pdf','.svg','.png']}}
 (OUT/'droid_forecast_setup.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({'figure':'droid_forecast_setup','geometry':review,'native_cells':native.get('unique_cells')}))

if __name__=='__main__':main()
