"""PyMuPDF stand-in for the skill's inspect_figure.py (poppler absent). For each figure: size, 100-ppi proof at print
size, 300-ppi crops (left/right halves), smallest font size at print width, words off the page, embedded fonts."""
import json
from pathlib import Path
import pymupdf
HERE = Path(__file__).resolve().parent
FIGS = {"fig_plugin": 2.7, "fig_qualitative": 5.5, "fig_tradeoff": 2.7}
rep = {}
for name, wid in FIGS.items():
    d = pymupdf.open(HERE / f"{name}.pdf"); p = d[0]; r = p.rect
    z = wid * 72 / r.width
    p.get_pixmap(matrix=pymupdf.Matrix(z * 100 / 72, z * 100 / 72)).save(HERE / f"qa/{name}_paper_100ppi.png")
    for cn, (x0, x1) in {"left": (0, .5), "right": (.5, 1)}.items():
        clip = pymupdf.Rect(r.x0 + x0 * r.width, r.y0, r.x0 + x1 * r.width, r.y1)
        p.get_pixmap(matrix=pymupdf.Matrix(z * 300 / 72, z * 300 / 72), clip=clip).save(HERE / f"qa/{name}_{cn}_300ppi.png")
    sizes = [s["size"] * z for b in p.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    off = [w[4] for w in p.get_text("words") if w[0] < r.x0 - .5 or w[2] > r.x1 + .5 or w[1] < r.y0 - .5 or w[3] > r.y1 + .5]
    rep[name] = {"width_in": round(r.width / 72 * z, 3), "height_in": round(r.height / 72 * z, 3),
                 "min_font_pt": round(min(sizes), 2), "n_spans": len(sizes), "words_off_page": off,
                 "fonts": sorted({f[3] for f in p.get_fonts()}), "file_kb": round((HERE / f"{name}.pdf").stat().st_size / 1024)}
print(json.dumps(rep, indent=1))
(HERE / "qa/inspection.json").write_text(json.dumps(rep, indent=1))
