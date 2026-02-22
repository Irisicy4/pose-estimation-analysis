#!/usr/bin/env python3
"""
Recompute error_type for yolo8n entries using local prediction files + GT annotations.

Root cause in 2rinse_error_type.py:
  load_predictions() expects flat keypoints [x,y,v]*17 (len=51),
  but yolo8n stores nested [[x,y,v]]*17 (len=17), so all predictions were skipped.

Usage:
  1. Place yolo8n predictions in:   yolo8n_results/{dataset}/{sample_id}_keypoints.json
  2. Place GT annotations in:       gt_annotations/{dataset}/annotations.json
     (from server: /raid/charles/MLLM-as-a-Judge/pose/data/{dataset}/annotations.json)
  3. Run:  python check/recompute_yolo8n_with_gt.py

This script re-uses the same classification logic from 2rinse_error_type.py
(fallback path) with the format bug fixed.
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_JSON = REPO_ROOT / "pairwise_encoded_480.json"
YOLO8N_DIR = REPO_ROOT / "yolo8n_results"
GT_DIR = REPO_ROOT / "gt_annotations"

N_KP = 17
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])
ERROR_PRIORITY = {"swap": 0, "inversion": 1, "miss": 2, "jitter": 3, "good": 4}

# MPII 16-keypoint to COCO 17-keypoint mapping
# MPII: 0=r_ankle 1=r_knee 2=r_hip 3=l_hip 4=l_knee 5=l_ankle
#       6=pelvis 7=thorax 8=upper_neck 9=head_top
#       10=r_wrist 11=r_elbow 12=r_shoulder 13=l_shoulder 14=l_elbow 15=l_wrist
# COCO: 0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear 5=l_shoulder 6=r_shoulder
#       7=l_elbow 8=r_elbow 9=l_wrist 10=r_wrist 11=l_hip 12=r_hip
#       13=l_knee 14=r_knee 15=l_ankle 16=r_ankle
MPII_TO_COCO = {
    0: 16, 1: 14, 2: 12, 3: 11, 4: 13, 5: 15,
    10: 10, 11: 8, 12: 6, 13: 5, 14: 7, 15: 9,
}


def _mpii_to_coco_kpts(mpii_flat):
    """Convert MPII 16-keypoint flat list (48 values) to COCO 17x3 array."""
    mpii = np.array(mpii_flat, dtype=float).reshape(-1, 3)
    coco = np.zeros((17, 3), dtype=float)
    for mi, ci in MPII_TO_COCO.items():
        if mi < len(mpii):
            coco[ci] = mpii[mi]
    return coco


# ── GT loading ──────────────────────────────────────────────────────────

def load_gt_annotations(dataset_name):
    ann_path = GT_DIR / dataset_name / "annotations.json"
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        data = json.load(f)

    # COCO format: has "images" and "annotations" top-level keys
    if "images" in data and "annotations" in data:
        filename_to_id = {img["file_name"]: img["id"] for img in data["images"]}
        id_to_anns = {}
        for ann in data["annotations"]:
            id_to_anns.setdefault(ann["image_id"], []).append(ann)
        return {"format": "coco", "filename_to_id": filename_to_id,
                "id_to_anns": id_to_anns}

    # MPII/custom format: has "annotations" list with "image_name" and "persons"
    if "annotations" in data and isinstance(data["annotations"], list):
        image_to_persons = {}
        for ann in data["annotations"]:
            img_name = ann.get("image_name", "")
            img_id = ann.get("image_id", img_name)
            image_to_persons[img_name] = ann.get("persons", [])
            image_to_persons[img_id] = ann.get("persons", [])
            stem = Path(img_name).stem
            image_to_persons[stem] = ann.get("persons", [])
            image_to_persons[stem + ".jpg"] = ann.get("persons", [])
        kp_format = data.get("keypoint_format", "")
        return {"format": "mpii", "image_to_persons": image_to_persons,
                "keypoint_format": kp_format}

    return None


def get_all_gt_for_image(gt_data, image_name, sample_id):
    if gt_data is None:
        return []

    fmt = gt_data.get("format", "coco")

    if fmt == "coco":
        fmap = gt_data["filename_to_id"]
        img_id = (fmap.get(image_name) or fmap.get(sample_id + ".jpg")
                  or fmap.get(sample_id))
        if img_id is None:
            return []
        all_gts = []
        for ann in gt_data["id_to_anns"].get(img_id, []):
            kp = ann.get("keypoints", [])
            bbox = ann.get("bbox")
            if len(kp) >= 51 and ann.get("num_keypoints", 0) > 0:
                kpts = np.array(kp[:51], dtype=float).reshape(17, 3)
                all_gts.append({"keypoints": kpts, "bbox": bbox})
        return all_gts

    if fmt == "mpii":
        imap = gt_data["image_to_persons"]
        persons = (imap.get(image_name) or imap.get(sample_id + ".jpg")
                   or imap.get(sample_id) or [])
        all_gts = []
        for p in persons:
            kp = p.get("keypoints", [])
            bbox = p.get("bbox")
            nkp = p.get("num_keypoints", 0)
            if len(kp) < 48 or nkp <= 0:
                continue
            coco_kpts = _mpii_to_coco_kpts(kp)
            all_gts.append({"keypoints": coco_kpts, "bbox": bbox})
        return all_gts

    return []


# ── Prediction loading (FORMAT BUG FIXED) ───────────────────────────────

def load_yolo8n_predictions(dataset, sample_id):
    path = YOLO8N_DIR / dataset / f"{sample_id}_keypoints.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    out = []
    for person in data.get("keypoints", []):
        kp = person.get("keypoints", [])
        bbox = person.get("bbox")
        # FIX: handle both nested [[x,y,v]*17] and flat [x,y,v]*17 formats
        if isinstance(kp, list) and len(kp) > 0 and isinstance(kp[0], list):
            flat = []
            for pt in kp:
                flat.extend(pt)
            kp = flat
        if len(kp) < 51:
            continue
        kpts = np.array(kp[:51], dtype=float).reshape(17, 3)
        out.append({"keypoints": kpts, "bbox": bbox})
    return out


# ── OKS & error classification (same as 2rinse_error_type.py fallback) ──

def get_bbox_area(bbox):
    if bbox is None or len(bbox) < 4:
        return 0.0
    return float(bbox[2] * bbox[3])


def _compute_ks(pred_xy, gt_xy, area, sigma):
    if area <= 0 or sigma <= 0:
        return 0.0
    d2 = (pred_xy[0] - gt_xy[0]) ** 2 + (pred_xy[1] - gt_xy[1]) ** 2
    return float(np.exp(-d2 / (2 * area * sigma ** 2)))


def _compute_oks(pred_kpts, gt_kpts, gt_bbox):
    pred_kpts = np.asarray(pred_kpts).reshape(-1, 3)
    gt_kpts = np.asarray(gt_kpts).reshape(-1, 3)
    n = min(len(pred_kpts), len(gt_kpts), len(COCO_SIGMAS))
    area = get_bbox_area(gt_bbox)
    if area <= 0:
        valid = gt_kpts[gt_kpts[:, 2] > 0]
        if len(valid) == 0:
            return 0.0
        area = (valid[:, 0].max() - valid[:, 0].min() + 1) * \
               (valid[:, 1].max() - valid[:, 1].min() + 1)
    if area <= 0:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        if gt_kpts[i, 2] <= 0:
            continue
        total += _compute_ks(pred_kpts[i, :2], gt_kpts[i, :2], area, COCO_SIGMAS[i])
        count += 1
    return total / count if count else 0.0


def _match_predictions_to_gt(preds, all_gts):
    if not all_gts:
        return []
    used_gt = set()
    pred_order = sorted(range(len(preds)),
                        key=lambda i: -np.mean(preds[i]["keypoints"][:, 2]))
    matches = []
    for pi in pred_order:
        best_gt, best_oks = None, -1.0
        for gi, gt in enumerate(all_gts):
            if gi in used_gt:
                continue
            oks = _compute_oks(preds[pi]["keypoints"], gt["keypoints"], gt["bbox"])
            if oks > best_oks:
                best_oks, best_gt = oks, gi
        if best_gt is not None:
            used_gt.add(best_gt)
            matches.append((pi, best_gt))
        else:
            matches.append((pi, None))
    return matches


def _classify_keypoint_error(pred_xy, gt_person, all_gts, gt_idx, kp_idx, area, sigma):
    gt_kpts = gt_person["keypoints"]
    if gt_kpts[kp_idx, 2] <= 0:
        return "good"
    ks_correct = _compute_ks(pred_xy, gt_kpts[kp_idx, :2], area, sigma)
    if ks_correct >= 0.85:
        return "good"
    if 0.5 <= ks_correct < 0.85:
        return "jitter"
    for j in range(N_KP):
        if j == kp_idx or gt_kpts[j, 2] <= 0:
            continue
        if _compute_ks(pred_xy, gt_kpts[j, :2], area, sigma) >= 0.5:
            return "inversion"
    for q, gtq in enumerate(all_gts):
        if q == gt_idx:
            continue
        area_q = get_bbox_area(gtq["bbox"]) or area
        for j in range(N_KP):
            if gtq["keypoints"][j, 2] <= 0:
                continue
            if _compute_ks(pred_xy, gtq["keypoints"][j, :2], area_q, COCO_SIGMAS[j]) >= 0.5:
                return "swap"
    return "miss"


def _person_error_type_fallback(pred_kpts, gt_person, all_gts, gt_idx):
    area = get_bbox_area(gt_person["bbox"])
    if area <= 0:
        v = gt_person["keypoints"][gt_person["keypoints"][:, 2] > 0]
        if len(v) > 0:
            area = (v[:, 0].max() - v[:, 0].min() + 1) * \
                   (v[:, 1].max() - v[:, 1].min() + 1)
    if area <= 0:
        return "jitter"
    errors = []
    for i in range(N_KP):
        if gt_person["keypoints"][i, 2] <= 0:
            continue
        e = _classify_keypoint_error(
            pred_kpts[i, :2], gt_person, all_gts, gt_idx, i, area, COCO_SIGMAS[i]
        )
        errors.append(e)
    if not errors:
        return "good"
    return Counter(errors).most_common(1)[0][0]


def _majority_error_type(per_person_types):
    if not per_person_types:
        return ""
    c = Counter(per_person_types)
    total = len(per_person_types)
    for typ, count in c.most_common():
        if count >= 0.8 * total:
            return typ
    return ""


def choice_level_error_type(per_person_types, n_pred, n_gt):
    if n_pred > n_gt:
        return "over instance"
    if n_pred < n_gt:
        return "less instance"
    return _majority_error_type(per_person_types)


# ── Main logic ──────────────────────────────────────────────────────────

def recompute_one_side(meta, side, gt_data):
    dataset = meta["dataset"]
    sample_id = meta["sample_id"]
    image_name = meta.get("image_name", "")

    preds = load_yolo8n_predictions(dataset, sample_id)
    all_gts = get_all_gt_for_image(gt_data, image_name, sample_id)
    n_pred = len(preds)
    n_gt = len(all_gts)

    if not all_gts:
        return None

    matches = _match_predictions_to_gt(preds, all_gts)
    per_person = []
    for pi, gi in matches:
        if gi is None:
            per_person.append("miss")
            continue
        per_person.append(
            _person_error_type_fallback(preds[pi]["keypoints"], all_gts[gi], all_gts, gi)
        )

    error_type = choice_level_error_type(per_person, n_pred, n_gt)

    meta[f"error_type_{side}"] = error_type
    meta[f"error_type_per_person_{side}"] = per_person
    meta[f"_yolo8n_n_pred_{side}"] = n_pred
    meta[f"_yolo8n_n_gt_{side}"] = n_gt

    return error_type


def main():
    if not GT_DIR.exists():
        print(f"ERROR: GT directory not found: {GT_DIR}")
        print(f"Please copy annotations from server:")
        print(f"  mkdir -p gt_annotations/{{coco300,cocowb300,mpii300}}")
        print(f"  scp server:/raid/charles/MLLM-as-a-Judge/pose/data/coco300/annotations.json gt_annotations/coco300/")
        print(f"  scp server:/raid/charles/MLLM-as-a-Judge/pose/data/cocowb300/annotations.json gt_annotations/cocowb300/")
        print(f"  scp server:/raid/charles/MLLM-as-a-Judge/pose/data/mpii300/annotations.json gt_annotations/mpii300/")
        sys.exit(1)

    gt_cache = {}
    gt_available = []
    gt_missing = []
    for ds in ("coco300", "cocowb300", "mpii300"):
        ann_path = GT_DIR / ds / "annotations.json"
        if ann_path.exists():
            gt_cache[ds] = load_gt_annotations(ds)
            gt_available.append(ds)
        else:
            gt_cache[ds] = None
            gt_missing.append(ds)

    print(f"GT available for:  {', '.join(gt_available) or 'none'}")
    if gt_missing:
        print(f"GT missing for:    {', '.join(gt_missing)}")
    print()

    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries.\n")

    backup_path = INPUT_JSON.with_suffix(
        f".pre_yolo8n_gt_fix_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(INPUT_JSON, backup_path)
    print(f"Backup: {backup_path}\n")

    stats = Counter()
    fix_log = []

    for entry in data:
        meta = entry["metadata"]
        dataset = meta.get("dataset", "")

        for side in ("a", "b"):
            if meta.get(f"model_{side}_id") != "yolo8n":
                continue

            gt_data = gt_cache.get(dataset)
            if gt_data is None:
                stats["skipped_no_gt"] += 1
                continue

            old_et = meta.get(f"error_type_{side}", "")
            old_pp = meta.get(f"error_type_per_person_{side}", [])

            new_et = recompute_one_side(meta, side, gt_data)
            if new_et is None:
                stats["skipped_no_gt_image"] += 1
                continue

            new_pp = meta.get(f"error_type_per_person_{side}", [])
            bpk = f"{meta['sample_id']}|{dataset}|{meta['model_a_id']}|{meta['model_b_id']}"
            n_pred = meta.get(f"_yolo8n_n_pred_{side}")
            n_gt = meta.get(f"_yolo8n_n_gt_{side}")

            if old_et != new_et or old_pp != new_pp:
                stats["changed"] += 1
                fix_log.append(
                    f"[{bpk}] enc={meta['encoding']}: error_type_{side} "
                    f"'{old_et}' → '{new_et}' "
                    f"(n_pred={n_pred}, n_gt={n_gt}, pp={new_pp})"
                )
            else:
                stats["unchanged"] += 1
            stats[f"new_type_{new_et}"] += 1

    # Fix instance_type if per_person list grew
    inst_fixes = 0
    for entry in data:
        meta = entry["metadata"]
        pp_a = meta.get("error_type_per_person_a", [])
        pp_b = meta.get("error_type_per_person_b", [])
        if meta.get("instance_type") == "1C1I" and (len(pp_a) > 1 or len(pp_b) > 1):
            meta["instance_type"] = "1CnI"
            inst_fixes += 1

    # Sync synthetic_error_type
    syn_fixes = 0
    for entry in data:
        meta = entry["metadata"]
        if not meta.get("synthetic") or not meta.get("synthetic_error_type"):
            continue
        syn_type = meta["synthetic_error_type"]
        et_a = meta.get("error_type_a", "")
        et_b = meta.get("error_type_b", "")
        match = (et_a == syn_type or et_b == syn_type or
                 (et_a == "good" and et_b == syn_type) or
                 (et_b == "good" and et_a == syn_type))
        if not match:
            non_good = [t for t in (et_a, et_b) if t and t != "good"]
            new_syn = non_good[0] if non_good else ""
            meta["synthetic_error_type"] = new_syn
            if not new_syn:
                meta["synthetic"] = False
            syn_fixes += 1

    print("Recomputation stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  instance_type fixes: {inst_fixes}")
    print(f"  synthetic_error_type synced: {syn_fixes}")
    print()

    with open(INPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {INPUT_JSON}")

    log_path = SCRIPT_DIR / "recompute_yolo8n_gt_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Recomputed at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Backup: {backup_path}\n")
        f.write(f"GT available: {', '.join(gt_available)}\n")
        f.write(f"GT missing: {', '.join(gt_missing)}\n\n")
        for k, v in sorted(stats.items()):
            f.write(f"{k}: {v}\n")
        f.write(f"instance_type fixes: {inst_fixes}\n")
        f.write(f"synthetic_error_type synced: {syn_fixes}\n")
        f.write(f"\nDetailed changes:\n\n")
        for line in fix_log:
            f.write(line + "\n")
    print(f"Log: {log_path}")
    print("\nDone. Run validate_pairwise.py to verify.")


if __name__ == "__main__":
    main()
