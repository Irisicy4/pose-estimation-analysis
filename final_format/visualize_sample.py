#!/usr/bin/env python3
"""
Test images.json and annotations.json (keypoint format) by sampling from images,
resolving each image and its annotations, and drawing keypoints on the images.
Writes one output image per sample (original + GT + each model in a row) to
visualize_sample_output/.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES_JSON = ROOT / "images.json"
ANNOTATIONS_JSON = ROOT / "annotations.json"
OUTPUT_DIR = ROOT / "visualize_sample_output"
MAX_SAMPLES = 5

# COCO 17 keypoint skeleton
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
]

COLORS_BGR = [
    (0, 255, 0),   # green
    (0, 0, 255),   # red
    (255, 0, 0),   # blue
    (255, 0, 255), # purple
    (255, 255, 0), # cyan
    (255, 128, 0), # orange
]
CONF_THRESHOLD = 0.2


def draw_skeleton(img: np.ndarray, keypoints_flat: list[float], color: tuple, thickness: int = 2) -> None:
    """Draw one skeleton. keypoints_flat: [x,y,v]*17."""
    kpts = np.array(keypoints_flat, dtype=float).reshape(-1, 3)
    for i, (x, y, conf) in enumerate(kpts):
        if conf > CONF_THRESHOLD and x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
            cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), 1)
    for (idx1, idx2) in COCO_SKELETON:
        if idx1 >= len(kpts) or idx2 >= len(kpts):
            continue
        x1, y1, c1 = kpts[idx1]
        x2, y2, c2 = kpts[idx2]
        if c1 > CONF_THRESHOLD and c2 > CONF_THRESHOLD and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_annotation(img: np.ndarray, ann: dict, color: tuple) -> None:
    """Draw all skeletons in annotation['predictions'] onto img."""
    predictions = ann.get("predictions", [])
    if not predictions:
        return
    for skeleton in predictions:
        if len(skeleton) >= 51:
            draw_skeleton(img, skeleton[:51], color)


def main() -> None:
    if not IMAGES_JSON.exists():
        raise SystemExit(f"Missing {IMAGES_JSON}")
    if not ANNOTATIONS_JSON.exists():
        raise SystemExit(f"Missing {ANNOTATIONS_JSON}")

    with open(IMAGES_JSON, "r") as f:
        images_list = json.load(f)
    with open(ANNOTATIONS_JSON, "r") as f:
        annotations_list = json.load(f)

    image_id_to_record = {img["id"]: img for img in images_list}
    image_id_to_anns = {}
    for ann in annotations_list:
        iid = ann["image_id"]
        image_id_to_anns.setdefault(iid, []).append(ann)

    sample_ids = [img["id"] for img in images_list[:MAX_SAMPLES]]
    if not sample_ids:
        print("No images in images.json")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Testing {len(sample_ids)} sample images -> {OUTPUT_DIR}")

    for image_id in sample_ids:
        rec = image_id_to_record.get(image_id)
        if not rec:
            print(f"  Skip {image_id}: not in images.json")
            continue
        file_path = rec.get("file_path", "")
        # file_path is "keypoint/{dataset}/{filename}.jpg"; images live at final_images/keypoint/...
        full_path = ROOT / "final_images" / file_path
        if not full_path.exists():
            print(f"  Skip {image_id}: image file not found at {full_path}")
            continue

        img0 = cv2.imread(str(full_path))
        if img0 is None:
            print(f"  Skip {image_id}: failed to load image")
            continue

        anns = image_id_to_anns.get(image_id, [])
        # Order: original, then gt, then others
        gt_anns = [a for a in anns if a.get("model_name") == "gt"]
        other_anns = [a for a in anns if a.get("model_name") != "gt"]
        ordered = gt_anns + sorted(other_anns, key=lambda a: a.get("model_name", ""))

        n_panels = 1 + len(ordered)
        h, w = img0.shape[:2]
        margin = 8
        label_h = 40
        panel_w = w
        panel_h = h + label_h
        canvas_w = n_panels * (panel_w + margin) + margin
        canvas_h = panel_h + margin * 2
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

        # Original
        x0 = margin
        canvas[label_h : label_h + h, x0 : x0 + w] = img0
        cv2.putText(canvas, "original", (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        x0 += panel_w + margin

        for i, ann in enumerate(ordered):
            img_panel = img0.copy()
            model_name = ann.get("model_name", "?")
            error_type = ann.get("error_type", "")
            score = ann.get("final_score", 0)
            color = COLORS_BGR[i % len(COLORS_BGR)]
            draw_annotation(img_panel, ann, color)
            canvas[label_h : label_h + h, x0 : x0 + w] = img_panel
            label = f"{model_name} | {error_type} | {score:.3f}"
            cv2.putText(canvas, label, (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            x0 += panel_w + margin

        out_path = OUTPUT_DIR / f"{image_id.replace('/', '_')}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"  Wrote {out_path.name} ({len(ordered)} annotations)")

    print("Done.")


if __name__ == "__main__":
    main()
