"""Float audit for paper/iclr2027.pdf: page of each Figure/Table caption vs pages that reference it.
Flags unreferenced floats and floats far from their first reference. Usage: python scripts/v2/audit_floats.py"""
import collections
import re
import sys

import pymupdf

d = pymupdf.open(sys.argv[1] if len(sys.argv) > 1 else "paper/iclr2027.pdf")
end_main = next(i + 1 for i, p in enumerate(d) if "Limitations." in p.get_text())
cap, refs = {}, collections.defaultdict(list)
for i, p in enumerate(d):
    t = p.get_text()
    for m in re.finditer(r"^(Figure|Table) (\d+):", t, re.M):
        cap[(m.group(1)[:3].lower(), m.group(2))] = i + 1
    tt = re.sub(r"(Figure|Table) \d+:", "", t.replace("\n", " "))
    for m in re.finditer(r"(Fig(?:ure|\.)s?|Tables?|Tab\.)\s*(\d+)((?:\s*(?:,|and|–|-)\s*\d+)*)", tt):
        kind = "fig" if m.group(1).startswith("Fig") else "tab"
        nums = [m.group(2)] + re.findall(r"\d+", m.group(3))
        for a_, b_ in re.findall(r"(\d+)\s*[–-]\s*(\d+)", m.group(2) + m.group(3)):  # expand ranges "23–25"
            nums += [str(x) for x in range(int(a_) + 1, int(b_))]
        for n in nums:
            refs[(kind, n)].append(i + 1)
print(f"main text ends on page {end_main} of {len(d)}")
bad = 0
for (k, n), pg in sorted(cap.items(), key=lambda x: (x[0][0], int(x[0][1]))):
    rp = sorted(refs[(k, n)])
    same_part = [r for r in rp if (r <= end_main) == (pg <= end_main)]
    first = same_part[0] if same_part else None
    flag = ""
    if not rp:
        flag = "UNREFERENCED"
    elif first is None:
        flag = "only referenced from the other part (no local lead-in)"
    elif first > pg:
        flag = f"appears {first - pg}p before its first local reference"
    elif pg - first > 1:
        flag = f"appears {pg - first}p after its first local reference"
    bad += bool(flag)
    print(f"{k}{n:>3} p{pg:<3} refs {rp[:6]} {flag}")
print(f"{bad} flagged")
