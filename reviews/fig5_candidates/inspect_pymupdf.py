"""PyMuPDF stand-in for the skill's inspect_figure.py (poppler is not installed on this host).
Renders each candidate PDF at 5.5 in print width (100 ppi proof) and 300-ppi crops; checks word boxes stay on the page
and reports the smallest font size used in the PDF."""
import sys, json
import pymupdf
from pathlib import Path
HERE = Path(__file__).resolve().parent
crops = {"left": (0, 0, .5, 1), "right": (.5, 0, 1, 1), "centre": (.25, 0, .75, 1)}
rep = {}
for name in sys.argv[1:] or ["A", "B", "C", "D"]:
    d = pymupdf.open(HERE / f"cand_{name}.pdf"); p = d[0]; r = p.rect
    z = (5.5 if name == "D" else 2.31) * 72 / r.width
    p.get_pixmap(matrix=pymupdf.Matrix(z * 100 / 72, z * 100 / 72)).save(HERE / f"review/{name}_paper_100ppi.png")
    for cn, (x0, y0, x1, y1) in crops.items():
        clip = pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
        p.get_pixmap(matrix=pymupdf.Matrix(z * 300 / 72, z * 300 / 72), clip=clip).save(HERE / f"review/{name}_{cn}_300ppi.png")
    sizes = [s["size"] * z for b in p.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    off = [w[4] for w in p.get_text("words") if w[0] < r.x0 - .5 or w[2] > r.x1 + .5 or w[1] < r.y0 - .5 or w[3] > r.y1 + .5]
    rep[name] = {"width_in": round(r.width / 72 * z, 3), "height_in": r.height / 72 * z, "min_font_pt": round(min(sizes), 2),
                 "n_spans": len(sizes), "words_off_page": off}
print(json.dumps(rep, indent=1))
(HERE / "review/inspection.json").write_text(json.dumps(rep, indent=1))
