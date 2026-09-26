"""Appendix qualitative figure: the single best-ranked window per source, for datasets not shown elsewhere
(Language-Table forecasting; V-JEPA 2-AC plug-in). Reuses make_qual_best / make_qual_best_plugin on the first
(largest-gain) window of their caches (results/v2/analysis/qual_best, selection rule in scripts/v2/qual_select.py)."""
import json
import numpy as np
import make_figures as mf
import make_qual_best as qb
import make_qual_best_plugin as qp


class First:
    """npz view keeping only the first window of every per-window array."""
    def __init__(self, z):
        self.z, self.n, self.files = z, len(z["episode"]), z.files

    def __getitem__(self, k):
        v = self.z[k]
        return v[:1] if getattr(v, "ndim", 0) and len(v) == self.n and k != "arms" else v


def main():
    z, info = qb.load("language_table", qb.SRC)
    fig = qb.make("language_table", First(z), info)
    fig.savefig(mf.FIG / "qual_one_language_table.pdf"); fig.savefig(mf.FIG / "qual_one_language_table_preview.png", dpi=220)
    z = np.load(qp.SRC / "vjepa2ac_droid.npz"); info = json.loads((qp.SRC / "vjepa2ac_droid.json").read_text())
    fig = qp.vjepa_fig(First(z), info)
    fig.savefig(mf.FIG / "qual_one_vjepa2ac.pdf"); fig.savefig(mf.FIG / "qual_one_vjepa2ac_preview.png", dpi=220)
    z, info = qb.load("droid_cam2", qb.SRC)
    fig = qb.make("droid_cam2", First(z), info)
    fig.savefig(mf.FIG / "qual_one_droid_cam2.pdf"); fig.savefig(mf.FIG / "qual_one_droid_cam2_preview.png", dpi=220)
    print("wrote qual_one_language_table, qual_one_vjepa2ac, qual_one_droid_cam2")


if __name__ == "__main__":
    main()
