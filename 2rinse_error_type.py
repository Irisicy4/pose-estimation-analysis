#!/usr/bin/env python3
"""
Enrich selected_pairwise.json with error_type metadata using the official
Ronchi & Perona ICCV 2017 implementation (coco-analyze).

Uses coco-analyze/analysisAPI (and pycocotools there) for localization error
classification (good, jitter, inversion, swap, miss) per paper Sec. 3.1.
Choice level: "over instance" / "less instance" map to paper Sec. 3.3
(Background FP/FN). Per-person type = worst among keypoint labels; choice-level
= 80% majority or over/less instance.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# Reuse official coco-analyze implementation when available (run "make" in coco-analyze to build _mask)
KEYPOINT_ROOT = Path("/raid/icy/aggregate-mllm-as-a-judge/keypoint")
COCO_ANALYZE_ROOT = KEYPOINT_ROOT / "coco-analyze"
USE_COCO_ANALYZE = False
_COCO = _COCOanalyze = None
if COCO_ANALYZE_ROOT.exists():
    if str(COCO_ANALYZE_ROOT) not in sys.path:
        sys.path.insert(0, str(COCO_ANALYZE_ROOT))
    try:
        from pycocotools.coco import COCO as _COCO
        from pycocotools.cocoanalyze import COCOanalyze as _COCOanalyze
        USE_COCO_ANALYZE = True
    except Exception as e:
        print("coco-analyze could not be loaded (e.g. build with 'make' in coco-analyze):", e, file=sys.stderr)
        try:
            reply = input("Fall back to in-script implementation? [y/N]: ").strip().lower()
        except EOFError:
            reply = "n"
        if reply in ("y", "yes"):
            USE_COCO_ANALYZE = False
        else:
            print("Aborting. Build coco-analyze or run again and confirm fallback.", file=sys.stderr)
            sys.exit(1)

CHARLES_POSE_ROOT = Path("/raid/charles/MLLM-as-a-Judge/pose")
DATA_ROOT = CHARLES_POSE_ROOT / "data"
RESULTS_ROOT = CHARLES_POSE_ROOT / "results"
SELECTED_JSON = CHARLES_POSE_ROOT / "selected" / "selected_pairwise.json"
OUTPUT_JSON = KEYPOINT_ROOT / "selected_pairwise_error_type.json"

N_KP = 17
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])
# Worst-first order for aggregating per-person type from keypoint labels
ERROR_PRIORITY = {"swap": 0, "inversion": 1, "miss": 2, "jitter": 3, "good": 4}


# ---------- Fallback when coco-analyze is not built (paper-aligned logic) ----------
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
        area = (valid[:, 0].max() - valid[:, 0].min() + 1) * (valid[:, 1].max() - valid[:, 1].min() + 1)
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
    pred_order = sorted(range(len(preds)), key=lambda i: -np.mean(preds[i]["keypoints"][:, 2]))
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
            area = (v[:, 0].max() - v[:, 0].min() + 1) * (v[:, 1].max() - v[:, 1].min() + 1)
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


def get_bbox_area(bbox):
    if bbox is None or len(bbox) < 4:
        return 0.0
    return float(bbox[2] * bbox[3])


def load_gt_annotations(dataset_name):
    ann_path = DATA_ROOT / dataset_name / "annotations.json"
    if not ann_path.exists():
        return None
    with open(ann_path, "r") as f:
        data = json.load(f)
    if "images" not in data or "annotations" not in data:
        return None
    filename_to_id = {img["file_name"]: img["id"] for img in data["images"]}
    id_to_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        id_to_anns.setdefault(img_id, []).append(ann)
    return {"filename_to_id": filename_to_id, "id_to_anns": id_to_anns}


def get_all_gt_for_image(gt_data, image_name):
    if gt_data is None:
        return []
    img_id = gt_data["filename_to_id"].get(image_name)
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


def load_predictions(model_name, dataset_name, image_stem):
    path = RESULTS_ROOT / model_name / dataset_name / f"{image_stem}_keypoints.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = json.load(f)
    out = []
    for person in data.get("keypoints", []):
        kp = person.get("keypoints", [])
        bbox = person.get("bbox")
        # Handle nested [[x,y,v]*17] format (e.g. yolo8n) in addition to flat [x,y,v]*17
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


def get_img_id(gt_data, image_name, sample_id):
    """Resolve COCO image id for this image. Tries image_name and sample_id + '.jpg'."""
    if not gt_data:
        return None
    fid = gt_data.get("filename_to_id") or {}
    return fid.get(image_name) or fid.get(sample_id + ".jpg") or fid.get(sample_id)


def _kp_17x3_to_coco_flat(kpts):
    """Convert (17,3) array to COCO keypoints list [x,y,v]*17."""
    kpts = np.asarray(kpts).reshape(17, 3)
    out = []
    for i in range(17):
        out.extend([float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])])
    return out


def build_coco_gt_one_image(img_id, image_name, all_gts):
    """Build a COCO-format dataset (single image) for coco-analyze. all_gts: list of {keypoints (17,3), bbox}."""
    images = [{"id": img_id, "file_name": image_name}]
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    annotations = []
    for idx, g in enumerate(all_gts):
        kp = g["keypoints"]
        bbox = g.get("bbox")
        if bbox is None or len(bbox) < 4:
            xs = kp[kp[:, 2] > 0, 0]
            ys = kp[kp[:, 2] > 0, 1]
            if len(xs) == 0:
                area = 0.0
                bbox = [0, 0, 0, 0]
            else:
                area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
                bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        else:
            area = float(bbox[2] * bbox[3])
        nkp = int((np.asarray(kp)[:, 2] > 0).sum())
        ann_id = img_id * 10000 + idx + 1  # unique per image
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": _kp_17x3_to_coco_flat(kp),
            "num_keypoints": nkp,
            "bbox": [float(b) for b in bbox[:4]],
            "area": area,
            "iscrowd": 0,
        })
    coco = _COCO(None)
    coco.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco.createIndex()
    return coco


def build_dt_list(img_id, preds):
    """Build list of detection dicts for COCO loadRes (keypoints format). preds: list of {keypoints (17,3), bbox}."""
    out = []
    for p in preds:
        kp = np.asarray(p["keypoints"]).reshape(17, 3)
        score = float(np.mean(kp[:, 2])) if kp[:, 2].size else 0.0
        out.append({
            "image_id": img_id,
            "category_id": 1,
            "keypoints": _kp_17x3_to_coco_flat(kp),
            "score": score,
        })
    return out


def _person_label_from_coco_cdt(cdt):
    """Map coco-analyze corrected_dt entry (good, jitter, inversion, swap, miss lists) to one label (worst)."""
    worst_rank, worst_label = -1, "good"
    for label in ("good", "jitter", "inversion", "swap", "miss"):
        if label not in cdt:
            continue
        arr = np.asarray(cdt[label])
        if arr.sum() <= 0:
            continue
        rank = ERROR_PRIORITY.get(label, 5)
        if rank < worst_rank or worst_rank < 0:
            worst_rank = rank
            worst_label = label
    return worst_label


def _run_coco_analyze_for_model(coco_gt, dt_list, quiet=True):
    """
    Run COCOanalyze keypoint error analysis for one image / one model.
    Returns (per_person_types, n_pred, n_gt). per_person_types = list of "good"|"jitter"|"inversion"|"swap"|"miss".
    """
    if not dt_list:
        n_gt = len(coco_gt.getAnnIds())
        return [], 0, n_gt
    coco_dt = coco_gt.loadRes(dt_list)
    analyzer = _COCOanalyze(coco_gt, coco_dt)
    analyzer.params.imgIds = sorted(coco_gt.getImgIds())
    analyzer.params.catIds = sorted(coco_gt.getCatIds())
    analyzer.params.areaRng = [[32 ** 2, 1e5 ** 2]]
    analyzer.params.areaRngLbl = ["all"]
    if quiet:
        import io
        _saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        analyzer.analyze(check_kpts=True, check_scores=False, check_bckgd=False)
    finally:
        if quiet:
            sys.stdout, sys.stderr = _saved
    corrected = analyzer.corrected_dts.get("all", [])
    per_person_types = [_person_label_from_coco_cdt(c) for c in corrected]
    n_pred = len(dt_list)
    n_gt = len(coco_gt.getAnnIds())
    return per_person_types, n_pred, n_gt


def _majority_error_type(per_person_types):
    """Return error type that >= 80% of instances share, else ''."""
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


def compute_instance_recall(n_pred, n_gt):
    """Fraction of GT instances detected. 1.0 when n_pred >= n_gt, <1.0 when missing."""
    if n_gt <= 0:
        return 1.0
    return min(n_pred, n_gt) / n_gt


def compute_adjusted_score(oks, n_pred, n_gt):
    """OKS penalized by instance recall: adjusted = OKS * min(n_pred/n_gt, 1.0).

    This addresses the limitation where OKS only measures quality of detected
    keypoints but does not penalize missing instances.  A model that detects
    3/10 persons accurately gets adjusted = OKS * 0.3 instead of raw OKS.
    """
    recall = compute_instance_recall(n_pred, n_gt)
    return oks * recall


def process_entry(entry, gt_cache):
    meta = entry.get("metadata", {})
    dataset = meta.get("dataset", "")
    image_name = meta.get("image_name", "")
    sample_id = meta.get("sample_id") or Path(image_name).stem
    model_a_id = meta.get("model_a_id", "")
    model_b_id = meta.get("model_b_id", "")

    if dataset not in gt_cache:
        gt_cache[dataset] = load_gt_annotations(dataset)
    gt_data = gt_cache[dataset]
    all_gts = get_all_gt_for_image(gt_data, image_name) if gt_data else []
    n_gt = len(all_gts)

    preds_a = load_predictions(model_a_id, dataset, sample_id)
    preds_b = load_predictions(model_b_id, dataset, sample_id)

    if USE_COCO_ANALYZE and _COCO is not None and _COCOanalyze is not None:
        img_id = get_img_id(gt_data, image_name, sample_id) if gt_data else None
        if img_id is None:
            img_id = 1
        coco_gt = build_coco_gt_one_image(img_id, image_name or (sample_id + ".jpg"), all_gts)
        dt_list_a = build_dt_list(img_id, preds_a)
        dt_list_b = build_dt_list(img_id, preds_b)
        per_person_a, n_pred_a, _ = _run_coco_analyze_for_model(coco_gt, dt_list_a)
        per_person_b, n_pred_b, _ = _run_coco_analyze_for_model(coco_gt, dt_list_b)
    else:
        matches_a = _match_predictions_to_gt(preds_a, all_gts)
        matches_b = _match_predictions_to_gt(preds_b, all_gts)
        per_person_a = []
        for pi, gi in matches_a:
            if gi is None:
                per_person_a.append("miss")
                continue
            per_person_a.append(_person_error_type_fallback(preds_a[pi]["keypoints"], all_gts[gi], all_gts, gi))
        per_person_b = []
        for pi, gi in matches_b:
            if gi is None:
                per_person_b.append("miss")
                continue
            per_person_b.append(_person_error_type_fallback(preds_b[pi]["keypoints"], all_gts[gi], all_gts, gi))
        n_pred_a, n_pred_b = len(preds_a), len(preds_b)

    error_type_a = choice_level_error_type(per_person_a, n_pred_a, n_gt)
    error_type_b = choice_level_error_type(per_person_b, n_pred_b, n_gt)

    # If both are same over/less instance, they likely predict the same set of GT; use majority rule instead
    if (error_type_a, error_type_b) in (("less instance", "less instance"), ("over instance", "over instance")):
        maj_a = _majority_error_type(per_person_a)
        maj_b = _majority_error_type(per_person_b)
        if maj_a:
            error_type_a = maj_a
        if maj_b:
            error_type_b = maj_b

    meta["error_type_a"] = error_type_a
    meta["error_type_b"] = error_type_b
    meta["error_type_per_person_a"] = per_person_a
    meta["error_type_per_person_b"] = per_person_b

    # Instance-aware metrics
    meta["n_pred_a"] = n_pred_a
    meta["n_pred_b"] = n_pred_b
    meta["n_gt"] = n_gt
    meta["instance_recall_a"] = compute_instance_recall(n_pred_a, n_gt)
    meta["instance_recall_b"] = compute_instance_recall(n_pred_b, n_gt)

    score_a = meta.get("score_a")
    score_b = meta.get("score_b")
    if score_a is not None:
        meta["adjusted_score_a"] = round(compute_adjusted_score(score_a, n_pred_a, n_gt), 6)
    if score_b is not None:
        meta["adjusted_score_b"] = round(compute_adjusted_score(score_b, n_pred_b, n_gt), 6)

    return entry


def main():
    if USE_COCO_ANALYZE:
        print("Using official coco-analyze (Ronchi et al. ICCV 2017) for error types.")
    else:
        print("Using fallback in-script implementation (build coco-analyze with 'make' for official logic).")
    with open(SELECTED_JSON, "r") as f:
        data = json.load(f)
    gt_cache = {}
    for i, entry in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(data)}")
        process_entry(entry, gt_cache)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} entries to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
