#!/usr/bin/env python3
"""
Sample pairwise pose comparisons from selected_pairwise_error_type.json:
- 180 by error type: (error_type_a == error_type_b) or one is 'good' and the other a type;
  30 per type (jitter, miss, inversion, swap, over instance, less instance); metadata synthetic=true, synthetic_error_type=<type>.
- 300 random; no synthetic flag.
Rebase on error-type JSON, enrich, permute A/B, produce 4 encoding variants (a,b,c,d).
"""

import os
import json
import random
import copy
from pathlib import Path

import cv2
import numpy as np

# Paths
CHARLES_POSE_ROOT = Path("/raid/charles/MLLM-as-a-Judge/pose")
ICY_KEYPOINT_ROOT = Path("/raid/icy/aggregate-mllm-as-a-judge/keypoint")
SELECTED_JSON = ICY_KEYPOINT_ROOT / "selected_pairwise_error_type.json"
CHARLES_METADATA_DIR = CHARLES_POSE_ROOT / "metadata"
CHARLES_RESULTS_DIR = CHARLES_POSE_ROOT / "results"
CHARLES_SELECTED_DIR = CHARLES_POSE_ROOT / "selected"

MEDIA_ROOT = ICY_KEYPOINT_ROOT / "media"
OUTPUT_JSON = ICY_KEYPOINT_ROOT / "pairwise_encoded_480.json"

SEED = 42
# Error types we sample 30 each (180 total)
ERROR_TYPES = ["jitter", "miss", "inversion", "swap", "over instance", "less instance"]
PER_TYPE_COUNT = 30
RANDOM_COUNT = 300

# Fixed prompt prefix/suffix (middle A/B part varies for encoding d)
PROMPT_PREFIX = "This is a human pose estimation task. Given the original image <image>, which pose estimation result is better?.\n"
PROMPT_SUFFIX = "\nPlease answer with A or B directly."

# COCO 17 skeleton and keypoint names
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
]
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COLORS_BGR = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (255, 0, 255),  # purple
    (255, 255, 0),  # cyan
    (255, 128, 0),  # orange
    (128, 0, 255),  # violet
    (0, 255, 255),  # yellow
]
# Per-limb colors for encode (c) - one color per skeleton edge
LIMB_COLORS = [
    (0, 255, 0), (0, 200, 0), (0, 150, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (200, 200, 0), (200, 0, 200), (0, 200, 200), (100, 100, 255), (100, 255, 100), (255, 100, 100),
    (180, 180, 0), (180, 0, 180), (0, 180, 180),
]


def load_gt_instance_count_lookup():
    """
    Build (dataset, sample_id) -> num_gt_instances from ground-truth annotations.
    This is the number of pose instances (persons) in the image, not the pairwise-matched count.
    """
    data_root = CHARLES_POSE_ROOT / "data"
    lookup = {}
    for dataset_dir in data_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        ann_path = dataset_dir / "annotations.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r") as f:
            data = json.load(f)
        # COCO-style: images[] with file_name, annotations[] with image_id
        if "images" in data and "annotations" in data:
            filename_to_id = {img["file_name"]: img["id"] for img in data["images"]}
            id_to_anns = {}
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                id_to_anns.setdefault(img_id, []).append(ann)
            dataset_name = dataset_dir.name
            for file_name, img_id in filename_to_id.items():
                stem = Path(file_name).stem
                n = len(id_to_anns.get(img_id, []))
                lookup[(dataset_name, stem)] = n
        # Fallback: other formats could be added here
    return lookup


def load_keypoints(model_id: str, dataset: str, sample_id: str):
    """Load keypoints JSON for one model/image. Returns list of persons, each person has 'keypoints' (17,3)."""
    path = CHARLES_RESULTS_DIR / model_id / dataset / f"{sample_id}_keypoints.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    persons = []
    for p in data.get("keypoints", []):
        kpts = p.get("keypoints", [])
        if len(kpts) >= 51:
            arr = np.array(kpts[:51], dtype=float).reshape(17, 3)
            persons.append(arr)
        elif len(kpts) == 17 * 3:
            persons.append(np.array(kpts, dtype=float).reshape(17, 3))
    return persons if persons else None


def draw_skeleton(img, keypoints, color, thickness=2, conf_threshold=0.2):
    keypoints = np.array(keypoints).reshape(-1, 3)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_threshold and x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
            cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), 1)
    for (idx1, idx2) in COCO_SKELETON:
        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue
        x1, y1, c1 = keypoints[idx1]
        x2, y2, c2 = keypoints[idx2]
        if c1 > conf_threshold and c2 > conf_threshold and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_skeleton_per_limb(img, keypoints, conf_threshold=0.2):
    keypoints = np.array(keypoints).reshape(-1, 3)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_threshold and x > 0 and y > 0:
            c = LIMB_COLORS[min(i, len(LIMB_COLORS) - 1)]
            cv2.circle(img, (int(x), int(y)), 4, c, -1)
            cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), 1)
    for ei, (idx1, idx2) in enumerate(COCO_SKELETON):
        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue
        x1, y1, c1 = keypoints[idx1]
        x2, y2, c2 = keypoints[idx2]
        if c1 > conf_threshold and c2 > conf_threshold and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            color = LIMB_COLORS[min(ei, len(LIMB_COLORS) - 1)]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def keypoints_to_json_text(persons):
    """Format list of (17,3) arrays as compact text for encoding (d). One line per person, \\n only between persons."""
    lines = []
    for i, kpts in enumerate(persons):
        kpts = np.asarray(kpts).reshape(17, 3)
        kp_list = []
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y = float(kpts[j, 0]), float(kpts[j, 1])
            kp_list.append({"name": name, "x": round(x, 1), "y": round(y, 1)})
        person_obj = {"person_id": i, "keypoints": kp_list}
        lines.append(json.dumps(person_obj, separators=(",", ":")))
    return "\n".join(lines)[:2000]  # cap length


def render_encoding_a(original_path: Path, persons_a, persons_b, out_path_a: Path, out_path_b: Path):
    """All predictions same color (green)."""
    img = cv2.imread(str(original_path))
    if img is None:
        return False
    green = (0, 255, 0)
    img_a = img.copy()
    img_b = img.copy()
    if persons_a:
        for kpts in persons_a:
            draw_skeleton(img_a, kpts, green)
    if persons_b:
        for kpts in persons_b:
            draw_skeleton(img_b, kpts, green)
    cv2.imwrite(str(out_path_a), img_a)
    cv2.imwrite(str(out_path_b), img_b)
    return True


def render_encoding_b(original_path: Path, persons_a, persons_b, out_path_a: Path, out_path_b: Path):
    """Each prediction (person) different color."""
    img = cv2.imread(str(original_path))
    if img is None:
        return False
    img_a = img.copy()
    img_b = img.copy()
    if persons_a:
        for i, kpts in enumerate(persons_a):
            draw_skeleton(img_a, kpts, COLORS_BGR[i % len(COLORS_BGR)])
    if persons_b:
        for i, kpts in enumerate(persons_b):
            draw_skeleton(img_b, kpts, COLORS_BGR[i % len(COLORS_BGR)])
    cv2.imwrite(str(out_path_a), img_a)
    cv2.imwrite(str(out_path_b), img_b)
    return True


def render_encoding_c(original_path: Path, persons_a, persons_b, out_path_a: Path, out_path_b: Path):
    """Different color per body part within each pose."""
    img = cv2.imread(str(original_path))
    if img is None:
        return False
    img_a = img.copy()
    img_b = img.copy()
    if persons_a:
        for kpts in persons_a:
            draw_skeleton_per_limb(img_a, kpts)
    if persons_b:
        for kpts in persons_b:
            draw_skeleton_per_limb(img_b, kpts)
    cv2.imwrite(str(out_path_a), img_a)
    cv2.imwrite(str(out_path_b), img_b)
    return True


def _synthetic_error_type_for_entry(entry):
    """Return synthetic_error_type if entry qualifies (same type or one good), else None."""
    meta = entry.get("metadata", {})
    ta = meta.get("error_type_a", "")
    tb = meta.get("error_type_b", "")
    if ta == tb:
        return ta if ta in ERROR_TYPES else None  # same type (only count non-good)
    if ta == "good" and tb in ERROR_TYPES:
        return tb
    if tb == "good" and ta in ERROR_TYPES:
        return ta
    return None


def main():
    random.seed(SEED)
    with open(SELECTED_JSON, "r") as f:
        all_entries = json.load(f)

    # 1) Build list of entries that qualify for by-type sampling (same type or one good)
    by_type = {t: [] for t in ERROR_TYPES}
    for entry in all_entries:
        typ = _synthetic_error_type_for_entry(entry)
        if typ is not None:
            by_type[typ].append(entry)

    # 2) Sample 30 per type
    synthetic_sampled = []
    per_type_counts = {}
    for typ in ERROR_TYPES:
        pool = by_type[typ]
        n = min(PER_TYPE_COUNT, len(pool))
        if n > 0:
            chosen = random.sample(pool, n)
            for e in chosen:
                ec = copy.deepcopy(e)
                ec["_synthetic_type"] = typ
                synthetic_sampled.append(ec)
            per_type_counts[typ] = n
        else:
            per_type_counts[typ] = 0
    print("Synthetic samples per error type:", per_type_counts)
    print("Total synthetic:", sum(per_type_counts.values()))
    # 3) 300 random from entries not in synthetic set
    def entry_key(ent):
        m = ent.get("metadata", {})
        return (m.get("dataset"), m.get("sample_id"), m.get("model_a_id"), m.get("model_b_id"))
    synthetic_keys = {entry_key(e) for e in synthetic_sampled}
    rest = [e for e in all_entries if entry_key(e) not in synthetic_keys]
    random_pool = random.sample(rest, min(RANDOM_COUNT, len(rest))) if rest else []
    print("Random samples:", len(random_pool))
    for e in random_pool:
        ec = copy.deepcopy(e)
        ec["_synthetic_type"] = None
        synthetic_sampled.append(ec)
    sampled = synthetic_sampled

    gt_instance_lookup = load_gt_instance_count_lookup()

    # Create output dirs
    enc_a_dir = MEDIA_ROOT / "encode_same_color"
    enc_b_dir = MEDIA_ROOT / "encode_per_person_color"
    enc_c_dir = MEDIA_ROOT / "encode_per_body_part"
    for d in (enc_a_dir, enc_b_dir, enc_c_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Resolve original image path: selected uses "images/coco300_000000319607.jpg" etc.
    def get_original_path(entry):
        rel = entry["metadata"].get("original_image") or entry["media"][0]
        if isinstance(rel, str) and not os.path.isabs(rel):
            return CHARLES_SELECTED_DIR / rel
        return Path(rel) if isinstance(rel, str) else None

    output_entries = []
    global_id = 0
    prompt_image_part = "A. <image>\nB. <image>"

    for idx, raw in enumerate(sampled):
        entry = copy.deepcopy(raw)
        meta = entry["metadata"]
        # Synthetic set: add synthetic=True and synthetic_error_type
        syn_type = entry.pop("_synthetic_type", None)
        meta["synthetic"] = syn_type is not None
        if syn_type is not None:
            meta["synthetic_error_type"] = syn_type
        dataset = meta.get("dataset", "")
        sample_id = meta.get("sample_id") or (Path(meta.get("image_name", "")).stem if meta.get("image_name") else "")
        model_a_id = meta.get("model_a_id", "")
        model_b_id = meta.get("model_b_id", "")

        # 1) Rename oks_a/oks_b -> score_a/score_b
        if "oks_a" in meta:
            meta["score_a"] = meta.pop("oks_a")
        if "oks_b" in meta:
            meta["score_b"] = meta.pop("oks_b")

        # 2) instance_type from GT annotation count (persons in this image), not pairwise metadata
        key_gt = (dataset, sample_id)
        n_inst = gt_instance_lookup.get(key_gt)
        if n_inst is None and sample_id:
            n_inst = gt_instance_lookup.get((dataset, Path(meta.get("image_name", "")).stem))
        n_inst = n_inst if n_inst is not None else 1
        meta["instance_type"] = "1C1I" if n_inst == 1 else "1CnI"

        # 3) Permute A/B 50%
        swap = random.random() < 0.5
        if swap:
            entry["media"] = [entry["media"][0], entry["media"][2], entry["media"][1]]
            entry["answer"] = "B" if entry["answer"] == "A" else "A"
            meta["vis_a"], meta["vis_b"] = meta.get("vis_b"), meta.get("vis_a")
            meta["score_a"], meta["score_b"] = meta.get("score_b"), meta.get("score_a")
            meta["model_a"], meta["model_b"] = meta.get("model_b"), meta.get("model_a")
            meta["model_a_id"], meta["model_b_id"] = meta.get("model_b_id"), meta.get("model_a_id")
            meta["better_model"], meta["worse_model"] = meta.get("worse_model"), meta.get("better_model")

        original_path = get_original_path(entry)
        stem = meta.get("image_name", f"{sample_id}.jpg").replace(".jpg", "")
        model_a_id = meta.get("model_a_id", "")
        model_b_id = meta.get("model_b_id", "")
        file_stem = f"{stem}_{model_a_id}_vs_{model_b_id}"

        # Keypoints for "current" A and B (after swap): choice A = first option, B = second
        persons_for_a = load_keypoints(model_a_id, dataset, sample_id) if original_path else None
        persons_for_b = load_keypoints(model_b_id, dataset, sample_id) if original_path else None
        if not persons_for_a:
            persons_for_a = []
        if not persons_for_b:
            persons_for_b = []
        # After swap, displayed A is old B and displayed B is old A
        if swap:
            persons_for_a, persons_for_b = persons_for_b, persons_for_a

        # 4) Duplicate 4 times: encodings a, b, c, d
        for enc_name, enc_key in [("same_color", "a"), ("per_person_color", "b"), ("per_body_part", "c"), ("json_text", "d")]:
            e = copy.deepcopy(entry)
            e["id"] = str(global_id)
            e["metadata"] = copy.deepcopy(meta)
            e["metadata"]["encoding"] = enc_key
            global_id += 1

            if enc_key == "a":
                rel_orig = entry["media"][0]
                out_a = enc_a_dir / f"{file_stem}_A.jpg"
                out_b = enc_a_dir / f"{file_stem}_B.jpg"
                if original_path and (persons_for_a or persons_for_b):
                    ok = render_encoding_a(original_path, persons_for_a, persons_for_b, out_a, out_b)
                    if ok:
                        e["media"] = [rel_orig, f"media/encode_same_color/{out_a.name}", f"media/encode_same_color/{out_b.name}"]
                    else:
                        e["media"] = entry["media"]
                else:
                    e["media"] = entry["media"]
                e["prompt"] = PROMPT_PREFIX + prompt_image_part + PROMPT_SUFFIX
            elif enc_key == "b":
                rel_orig = entry["media"][0]
                out_a = enc_b_dir / f"{file_stem}_A.jpg"
                out_b = enc_b_dir / f"{file_stem}_B.jpg"
                if original_path and (persons_for_a or persons_for_b):
                    ok = render_encoding_b(original_path, persons_for_a, persons_for_b, out_a, out_b)
                    if ok:
                        e["media"] = [rel_orig, f"media/encode_per_person_color/{out_a.name}", f"media/encode_per_person_color/{out_b.name}"]
                    else:
                        e["media"] = entry["media"]
                else:
                    e["media"] = entry["media"]
                e["prompt"] = PROMPT_PREFIX + prompt_image_part + PROMPT_SUFFIX
            elif enc_key == "c":
                rel_orig = entry["media"][0]
                out_a = enc_c_dir / f"{file_stem}_A.jpg"
                out_b = enc_c_dir / f"{file_stem}_B.jpg"
                if original_path and (persons_for_a or persons_for_b):
                    ok = render_encoding_c(original_path, persons_for_a, persons_for_b, out_a, out_b)
                    if ok:
                        e["media"] = [rel_orig, f"media/encode_per_body_part/{out_a.name}", f"media/encode_per_body_part/{out_b.name}"]
                    else:
                        e["media"] = entry["media"]
                else:
                    e["media"] = entry["media"]
                e["prompt"] = PROMPT_PREFIX + prompt_image_part + PROMPT_SUFFIX
            else:
                # d: JSON text for A and B, no choice images; keep single original in media
                json_a = keypoints_to_json_text(persons_for_a) if persons_for_a else "[]"
                json_b = keypoints_to_json_text(persons_for_b) if persons_for_b else "[]"
                rel_orig = entry["media"][0]
                e["prompt"] = PROMPT_PREFIX + f"A. {json_a}\nB. {json_b}" + PROMPT_SUFFIX
                e["media"] = [rel_orig]

            e["choices"] = ["A", "B"]
            output_entries.append(e)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_entries, f, indent=2)
    print(f"Wrote {len(output_entries)} entries to {OUTPUT_JSON}")
    print(f"Media subfolders: encode_same_color, encode_per_person_color, encode_per_body_part under {MEDIA_ROOT}")


if __name__ == "__main__":
    main()
