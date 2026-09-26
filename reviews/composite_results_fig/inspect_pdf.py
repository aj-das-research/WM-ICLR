"""PyMuPDF proof: 100-ppi print-size render, 300-ppi crops (3x2 grid for main, 4 cols for alt), min font, off-page words."""
import json, pymupdf
from pathlib import Path
H = Path(__file__).resolve().parent; rep = {}
for name, grid in (("composite_main", (3, 2)), ("composite_alt", (4, 1))):
    p = pymupdf.open(H / f"{name}.pdf")[0]; r = p.rect; z = 5.5 * 72 / r.width
    p.get_pixmap(matrix=pymupdf.Matrix(z * 100 / 72, z * 100 / 72)).save(H / f"qa/{name}_print_100ppi.png")
    nx, ny = grid
    for i in range(nx):
        for j in range(ny):
            clip = pymupdf.Rect(r.x0 + i * r.width / nx, r.y0 + j * r.height / ny, r.x0 + (i + 1) * r.width / nx, r.y0 + (j + 1) * r.height / ny)
            p.get_pixmap(matrix=pymupdf.Matrix(z * 300 / 72, z * 300 / 72), clip=clip).save(H / f"qa/{name}_crop_{j}{i}_300ppi.png")
    sizes = [s["size"] * z for b in p.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    off = [w[4] for w in p.get_text("words") if w[0] < r.x0 - .5 or w[2] > r.x1 + .5 or w[1] < r.y0 - .5 or w[3] > r.y1 + .5]
    rep[name] = {"size_in": [round(r.width / 72 * z, 3), round(r.height / 72 * z, 3)], "min_font_pt": round(min(sizes), 2), "words_off_page": off}
print(json.dumps(rep, indent=1)); (H / "qa/inspection.json").write_text(json.dumps(rep, indent=1))
