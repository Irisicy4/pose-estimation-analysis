#!/usr/bin/env python3
"""
Convert keypoint pose data (from selected_pairwise_error_type.json + original
pose estimations and GT) into the same format as ref instance_seg:
- images.json: original images (one record per unique image)
- annotations.json: one annotation per (image, model) or per (image, gt).
  predictions = list of all skeletons on that base image (each skeleton = [x,y,v]*17).
  error_type = image-level aggregated (good, jitter, miss, inversion, swap, over instance, less instance).
  final_score: GT=1.0; else avg_oks + 0.1*avg_oks, minus 0.1 per instance gap when type is less/over instance.

Uses logic and paths from 2rinse_error_type.py and rinse_pairwise_encode.py.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import numpy as np

# Paths (aligned with 2rinse_error_type.py and rinse_pairwise_encode.py)
KEYPOINT_ROOT = Path(__file__).resolve().parent.parent
CHARLES_POSE_ROOT = Path("/raid/charles/MLLM-as-a-Judge/pose")
DATA_ROOT = CHARLES_POSE_ROOT / "data"
RESULTS_ROOT = CHARLES_POSE_ROOT / "results"
SELECTED_JSON = KEYPOINT_ROOT / "selected_pairwise_error_type.json"
CHARLES_SELECTED_DIR = CHARLES_POSE_ROOT / "selected"

ROOT = Path(__file__).resolve().parent
FINAL_IMAGES_DIR = ROOT / "final_images" / "keypoint"
OUTPUT_IMAGES_JSON = ROOT / "images.json"
OUTPUT_ANNOTATIONS_JSON = ROOT / "annotations.json"


def load_gt_annotations(dataset_name: str) -> dict | None:
    """Load GT annotations and image list from COCO-format annotations.json.
    Returns dict with filename_to_id, id_to_anns, and id_to_image (for height/width).
    """
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
    id_to_image = {img["id"]: img for img in data["images"]}
    return {
        "filename_to_id": filename_to_id,
        "id_to_anns": id_to_anns,
        "id_to_image": id_to_image,
    }


def get_img_id(gt_data: dict | None, image_name: str, sample_id: str = "") -> int | None:
    """Resolve COCO image id. Tries image_name, sample_id + '.jpg', sample_id."""
    if not gt_data:
        return None
    fid = gt_data.get("filename_to_id") or {}
    return fid.get(image_name) or (fid.get(sample_id + ".jpg") if sample_id else None) or (fid.get(sample_id) if sample_id else None)


def get_gt_instance_count(gt_data: dict | None, image_name: str, sample_id: str = "") -> int:
    """Count ALL person annotations for this image (no keypoint filter). Use for n_gt."""
    if gt_data is None:
        return 0
    img_id = get_img_id(gt_data, image_name, sample_id)
    if img_id is None:
        return 0
    return len(gt_data["id_to_anns"].get(img_id, []))


def get_all_gt_for_image(gt_data: dict | None, image_name: str, sample_id: str = "") -> list[dict]:
    """Return list of GT instances with valid keypoints (51 values, num_keypoints > 0)."""
    if gt_data is None:
        return []
    img_id = get_img_id(gt_data, image_name, sample_id)
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


def get_image_dims(gt_data: dict | None, image_name: str, sample_id: str = "") -> tuple[int | None, int | None]:
    """Return (height, width) for image from GT if available."""
    if gt_data is None:
        return None, None
    img_id = get_img_id(gt_data, image_name, sample_id)
    if img_id is None:
        return None, None
    img = gt_data.get("id_to_image", {}).get(img_id)
    if img is None:
        return None, None
    return img.get("height"), img.get("width")


def load_predictions(model_name: str, dataset_name: str, image_stem: str) -> list[dict]:
    """Load predictions for one model on one image. Returns list of {keypoints (17,3), bbox}."""
    path = RESULTS_ROOT / model_name / dataset_name / f"{image_stem}_keypoints.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = json.load(f)
    out = []
    for person in data.get("keypoints", []):
        kp = person.get("keypoints", [])
        bbox = person.get("bbox")
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


def kp_17x3_to_flat(kpts: np.ndarray) -> list[float]:
    """Convert (17,3) array to COCO keypoints list [x,y,v]*17."""
    kpts = np.asarray(kpts).reshape(17, 3)
    out = []
    for i in range(17):
        out.extend([float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])])
    return out


def collect_image_model_metadata(entries: list[dict]) -> tuple[set[tuple[str, str]], dict]:
    """
    From pairwise entries, collect:
    - unique_images: set of (dataset, sample_id)
    - image_model_meta: (dataset, sample_id, model_id) -> {score (avg oks), error_type (image-level), n_pred, n_gt}
    Uses first occurrence when a (image, model) appears in multiple entries.
    """
    unique_images = set()
    image_model_meta = {}  # (dataset, sample_id, model_id) -> dict

    for entry in entries:
        meta = entry.get("metadata", {})
        dataset = meta.get("dataset", "")
        sample_id = meta.get("sample_id") or Path(meta.get("image_name", "")).stem
        if not dataset or not sample_id:
            continue
        unique_images.add((dataset, sample_id))
        n_gt = meta.get("n_gt")

        for role in ("a", "b"):
            model_id = meta.get("model_a_id" if role == "a" else "model_b_id", "")
            if not model_id:
                continue
            key = (dataset, sample_id, model_id)
            if key in image_model_meta:
                continue
            score = meta.get("score_a") or meta.get("oks_a") if role == "a" else meta.get("score_b") or meta.get("oks_b")
            error_type = meta.get("error_type_a") if role == "a" else meta.get("error_type_b")
            n_pred = meta.get("n_pred_a") if role == "a" else meta.get("n_pred_b")
            image_model_meta[key] = {
                "score": score,
                "error_type": (error_type or "").strip() or "good",
                "n_pred": n_pred,
                "n_gt": n_gt,
            }

    return unique_images, image_model_meta


def build_images(
    unique_images: set[tuple[str, str]],
    gt_cache: dict,
) -> list[dict]:
    """Build images list; copy originals to final_images/keypoint/{dataset}/."""
    FINAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    for dataset, sample_id in sorted(unique_images):
        image_id = f"{dataset}_{sample_id}"
        image_name = f"{sample_id}.jpg"
        # Resolve original path (same as rinse_pairwise_encode)
        rel = f"images/{dataset}_{sample_id}.jpg" if dataset else f"images/{sample_id}.jpg"
        src = CHARLES_SELECTED_DIR / rel
        if not src.exists():
            alt = CHARLES_SELECTED_DIR / "images" / image_name
            if alt.exists():
                src = alt
        ext = ".jpg"
        dst_dir = FINAL_IMAGES_DIR / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = f"{sample_id}{ext}"
        dst = dst_dir / dst_name
        if src.exists():
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
        file_path = f"keypoint/{dataset}/{dst_name}"

        gt_data = gt_cache.get(dataset)
        height, width = get_image_dims(gt_data, image_name, sample_id)
        if height is None and gt_data:
            height, width = get_image_dims(gt_data, f"{sample_id}.jpg", sample_id)

        images.append({
            "id": image_id,
            "file_path": file_path,
            "data_source": dataset or "pose",
            "height": height,
            "width": width,
            "scene": "",
            "is_crowd": False,
            "is_longtail": False,
            "groundtruth_class": ["person"],
            "groundtruth_class_id": [],
        })
    return images


def compute_final_score(avg_oks: float, error_type: str, n_pred: int | None, n_gt: int | None) -> float:
    """
    final_score: avg_oks + avg_oks*0.1; for less/over instance also subtract 0.1 per instance gap.
    Floor at 0.1*avg_oks to avoid 0: score = max(score, 0.1*avg_oks). Then clamp to [0, 1].
    """
    base = float(avg_oks) if avg_oks is not None else 0.0
    score = base + base * 0.1  # always add avg_oks * 0.1
    if error_type in ("less instance", "over instance") and n_pred is not None and n_gt is not None and n_gt > 0:
        gap = abs(n_pred - n_gt)
        score = score - 0.1 * gap
    score = max(score, base * 0.1)  # avoid 0; at least 0.1*avg_oks
    return round(max(0.0, min(1.0, score)), 6)


def instance_type_from_count(n: int) -> str:
    """1C1I if one person, else 1CnI."""
    return "1C1I" if n <= 1 else "1CnI"


# Same ±30% rule as 2rinse_error_type: less/over precedes other types
INSTANCE_RATIO_LOW = 0.7
INSTANCE_RATIO_HIGH = 1.3


def resolve_error_type_from_counts(
    n_pred: int,
    n_gt: int,
    fallback_error_type: str,
) -> str:
    """Recompute error_type from actual n_pred and n_gt so it matches written predictions."""
    n_gt = n_gt or 0
    if n_gt > 0:
        if n_pred > n_gt * INSTANCE_RATIO_HIGH:
            return "over instance"
        if n_pred < n_gt * INSTANCE_RATIO_LOW:
            return "less instance"
    elif n_pred > 0:
        return "over instance"
    return (fallback_error_type or "good").strip() or "good"


def build_annotations(
    unique_images: set[tuple[str, str]],
    image_model_meta: dict,
    gt_cache: dict,
) -> list[dict]:
    """
    One annotation per (image, gt) and one per (image, model).
    predictions = list of all skeletons on that base image (each skeleton = [x,y,v]*17).
    error_type = image-level (gt, good, jitter, miss, inversion, swap, over instance, less instance).
    final_score: GT=1.0; else 1.1*avg_oks - 0.1*gap when less/over instance, else 1.1*avg_oks.
    """
    annotations = []
    ann_counter = [0]

    def next_id():
        i = ann_counter[0]
        ann_counter[0] += 1
        return f"kp_{i}"

    for dataset, sample_id in sorted(unique_images):
        image_id = f"{dataset}_{sample_id}"
        image_name = f"{sample_id}.jpg"
        gt_data = gt_cache.get(dataset)
        all_gts = get_all_gt_for_image(gt_data, image_name, sample_id)
        n_gt = get_gt_instance_count(gt_data, image_name, sample_id) if gt_data else len(all_gts)

        # One GT annotation per image: all GT skeletons in one predictions list
        if all_gts:
            predictions_list = [kp_17x3_to_flat(g["keypoints"]) for g in all_gts]
            ann = {
                "id": next_id(),
                "task": "keypoint",
                "image_id": image_id,
                "coi": ["person"],
                "instance_type": instance_type_from_count(n_gt),
                "model_name": "gt",
                "error_type": "gt",
                "final_score": 1.0,
                "other_scores": {"oks": 1.0},
                "predictions_type": "keypoint",
                "predictions": predictions_list,
            }
            annotations.append(ann)

        # instance_type from GT count (difficulty of person as COI), not from current annotation
        image_instance_type = instance_type_from_count(n_gt)

        # One annotation per (image, model): all predicted skeletons in one predictions list
        models_on_image = [mid for (ds, sid, mid) in image_model_meta if (ds, sid) == (dataset, sample_id)]
        for model_id in sorted(models_on_image):
            meta = image_model_meta.get((dataset, sample_id, model_id), {})
            avg_oks = meta.get("score")
            if avg_oks is None:
                avg_oks = 0.0
            error_type = meta.get("error_type", "good")
            n_pred_meta = meta.get("n_pred")
            n_gt_meta = meta.get("n_gt")

            preds = load_predictions(model_id, dataset, sample_id)
            n_pred = len(preds)  # use actual count so error_type matches written predictions
            n_gt_val = n_gt_meta if n_gt_meta is not None else n_gt

            error_type = resolve_error_type_from_counts(n_pred, n_gt_val, meta.get("error_type", "good"))
            predictions_list = [kp_17x3_to_flat(p["keypoints"]) for p in preds]
            final_score = compute_final_score(avg_oks, error_type, n_pred, n_gt_val)

            ann = {
                "id": next_id(),
                "task": "keypoint",
                "image_id": image_id,
                "coi": ["person"],
                "instance_type": image_instance_type,
                "model_name": model_id,
                "error_type": error_type,
                "final_score": final_score,
                "other_scores": {"oks": float(avg_oks), "avg_oks": float(avg_oks)},
                "predictions_type": "keypoint",
                "predictions": predictions_list,
            }
            annotations.append(ann)

    return annotations


def main() -> None:
    if not SELECTED_JSON.exists():
        raise SystemExit(f"Missing {SELECTED_JSON}")

    with open(SELECTED_JSON, "r") as f:
        entries = json.load(f)

    unique_images, image_model_meta = collect_image_model_metadata(entries)
    print(f"Unique images: {len(unique_images)}")
    print(f"(image, model) meta entries: {len(image_model_meta)}")

    gt_cache = {}
    for dataset, _ in unique_images:
        if dataset not in gt_cache:
            gt_cache[dataset] = load_gt_annotations(dataset)

    images = build_images(unique_images, gt_cache)
    print(f"Built {len(images)} image records -> {FINAL_IMAGES_DIR}")

    annotations = build_annotations(unique_images, image_model_meta, gt_cache)
    print(f"Built {len(annotations)} annotations")

    with open(OUTPUT_IMAGES_JSON, "w") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_ANNOTATIONS_JSON, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_IMAGES_JSON} and {OUTPUT_ANNOTATIONS_JSON}")


if __name__ == "__main__":
    main()
