#!/usr/bin/env python3
"""Real, unaltered DROID100 ingestion frames; not a prediction/result figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/real_video/droid_100/schema_audit"
OUT = ROOT / "paper/generated/real_video"
CAMERAS = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["status"] != "complete", "This figure intentionally uses only schema-audit data"
    assert len(manifest["episodes"]) == 1
    row = manifest["episodes"][0]
    assert row["source_episode_index"] == 0
    assert row["extraction_identity"]["schema_audit_only"] is True
    indexes = None
    arrays, assets = {}, []
    for camera in CAMERAS:
        record = row["cameras"][camera]
        path = DATA / record["file"]
        assert sha(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as data:
            images, frames = data["images"], data["frame_indices"]
            if indexes is None:
                indexes = np.rint(np.linspace(0, len(images)-1, 3)).astype(int)
                native = frames[indexes].copy()
            else:
                np.testing.assert_array_equal(frames[indexes], native)
            assert images.dtype == np.uint8 and images.shape[1:] == (180, 320, 3)
            arrays[camera] = images[indexes].copy()
        for image, position, native_index in zip(arrays[camera], indexes, native):
            asset = OUT / "assets" / f"{camera}_native{native_index:03d}.png"
            Image.fromarray(image).save(asset)
            np.testing.assert_array_equal(np.array(Image.open(asset)), image)
            assets.append({"camera": camera, "stored_position": int(position), "native_frame_index": int(native_index),
                           "source_npz": str(path.relative_to(ROOT)), "source_npz_sha256": sha(path),
                           "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                           "asset": str(asset.relative_to(ROOT)), "asset_sha256": sha(asset),
                           "crop": "none", "color_adjustment": "none", "publisher_blur": "preserved",
                           "dimensions": list(image.shape)})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "image.composite_image": False})
    width, height = 5.5, 3.42
    fig = plt.figure(figsize=(width, height), dpi=200, facecolor="white")
    ink, muted = "#243447", "#536373"
    texts = []
    def text(x, y, value, size=8.5, color=ink, weight="normal", ha="left", va="center"):
        item = fig.text(x/width, y/height, value, fontsize=size, color=color,
                        fontweight=weight, ha=ha, va=va)
        texts.append(item)
        return item
    text(.05, 3.28, "One recording, three camera views", size=10.2, weight="bold")
    text(5.45, 3.28, "DROID-100", size=8, color=muted, ha="right")
    left, right, gap = .76, .04, .05
    panel_width = (width-left-right-2*gap)/3
    panel_height = panel_width*180/320
    tops = [2.91, 1.93, .95]
    image_axes = []
    for column, native_index in enumerate(native):
        x = left+column*(panel_width+gap)
        text(x+panel_width/2, 3.05, f"Frame {native_index}", size=8.5, ha="center", weight="bold")
    labels = [("External 1", "primary"), ("External 2", "transfer"), ("Wrist", "inspection")]
    for row_index, camera in enumerate(CAMERAS):
        bottom = tops[row_index]-panel_height
        mid = bottom+panel_height/2
        title, detail = labels[row_index]
        text(.055, mid+.045, title, size=8.2, weight="bold")
        text(.055, mid-.125, detail, size=8, color=muted)
        for column, frame in enumerate(arrays[camera]):
            x = left+column*(panel_width+gap)
            ax = fig.add_axes([x/width,bottom/height,panel_width/width,panel_height/height])
            ax.imshow(frame, interpolation="none", aspect="equal")
            ax.set_axis_off()
            border = Rectangle((0,0),1,1,transform=ax.transAxes,facecolor="none",edgecolor="#BCC4CC",linewidth=.55)
            ax.add_patch(border)
            image_axes.append(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    image_boxes = [ax.get_window_extent(renderer) for ax in image_axes]
    collisions = []
    text_boxes = []
    for item in texts:
        box = item.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            collisions.append({"type":"label_outside_canvas", "label":item.get_text()})
        for image_box in image_boxes:
            if box.overlaps(image_box):
                collisions.append({"type":"label_over_image", "label":item.get_text()})
        for earlier, other in text_boxes:
            if box.overlaps(other):
                collisions.append({"type":"label_overlap", "labels":[earlier,item.get_text()]})
        text_boxes.append((item.get_text(),box))
    assert not collisions, collisions
    basename = OUT / "recorded_droid_views"
    fig.savefig(basename.with_suffix(".pdf"), metadata={"Title":"Recorded DROID100 camera views: ingestion illustration", "Author":"ShiftWM research artifact"})
    fig.savefig(basename.with_suffix(".svg"))
    fig.savefig(basename.with_suffix(".png"), dpi=300)
    plt.close(fig)
    # Explicit grayscale inspection proof, never an experimental input.
    Image.open(basename.with_suffix(".png")).convert("L").save(OUT / "recorded_droid_views_grayscale.png")
    ledger = {"figure_type":"actual_data_ingestion_illustration_not_algorithm_result", "episode_id":row["episode_id"],
              "source_release":"official DROID100 debugging release", "source_shard":row["source_shard"],
              "source_episode_index":row["source_episode_index"],
              "source_manifest_sha256":sha(DATA/"manifest.json"), "source_data_audit_sha256":sha(DATA/"data_audit.json"),
              "selection_rule":"first schema-audited record; round(linspace(0,last_stored_frame,3)); no outcome-based selection",
              "native_frame_indices":native.tolist(), "stored_positions":indexes.tolist(),
              "planned_study_is_separate":"24 prespecified full-release shards, 1126 audited episodes; images shown here are DROID100 ingestion examples",
              "study_manifest_sha256":sha(ROOT/"data/real_video/droid_selected/processed/manifest.json"),
              "study_data_audit_sha256":sha(ROOT/"data/real_video/droid_selected/processed/data_audit.json"),
              "licence":"CC-BY-4.0; publisher face blur preserved", "generated_images":False,
              "action_projections":False, "time_units":"ordinal native frame index; no measured timestamps available",
              "width_inches":width, "height_inches":height, "minimum_text_points":min(t.get_fontsize() for t in texts),
              "layout_collisions":collisions,"image_panel_count":9,"asset_boundary":"observed raster images; editable vector text and borders",
              "assets":assets,"canonical_source":str(Path(__file__).relative_to(ROOT)), "source_sha256":sha(__file__),
              "protocol_sha256":sha(ROOT/"reports/real_droid_protocol.md"),
              "outputs":{str(basename.with_suffix(ext).relative_to(ROOT)):sha(basename.with_suffix(ext)) for ext in (".pdf",".svg",".png")}}
    (OUT/"recorded_droid_views_ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
    print(json.dumps({"figure":str(basename),"indices":native.tolist(),"collisions":collisions,"width_inches":width}))


if __name__ == "__main__":
    main()
