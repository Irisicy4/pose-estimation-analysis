#!/usr/bin/env python3
"""
Recompute error_type for yolo8n entries using locally available prediction files.

Root cause: 2rinse_error_type.py expected flat keypoints [x,y,v]*17 (len=51),
but yolo8n stores nested [[x,y,v]]*17 (len=17), so all predictions were skipped.

Without GT annotations locally, full per-keypoint classification (jitter/miss/
inversion/swap) is not possible. This script uses:
  - n_pred from yolo8n prediction files (fixed format)
  - n_gt inferred from the other model's data
  - OKS scores (already correctly computed in original pipeline)
to set a reasonable choice-level error_type.
"""

import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_JSON = REPO_ROOT / "pairwise_encoded_480.json"
YOLO8N_DIR = REPO_ROOT / "yolo8n_results"


def load_yolo8n_predictions(dataset, sample_id):
    path = YOLO8N_DIR / dataset / f"{sample_id}_keypoints.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    persons = []
    for p in data.get("keypoints", []):
        kp = p.get("keypoints", [])
        if isinstance(kp, list) and len(kp) > 0:
            if isinstance(kp[0], list):
                flat = []
                for pt in kp:
                    flat.extend(pt)
                kp = flat
            if len(kp) >= 51:
                arr = np.array(kp[:51], dtype=float).reshape(17, 3)
                persons.append(arr)
    return persons


def infer_n_gt(meta, yolo8n_side):
    """Best-effort inference of n_gt from available metadata."""
    other_side = "b" if yolo8n_side == "a" else "a"
    other_et = meta.get(f"error_type_{other_side}", "")
    other_pp = meta.get(f"error_type_per_person_{other_side}", [])
    n_other_pred = len(other_pp)

    if other_et == "good" and n_other_pred > 0:
        return n_other_pred
    if other_et in ("jitter", "miss", "inversion", "swap") and n_other_pred > 0:
        return n_other_pred
    if other_et == "less instance" and n_other_pred > 0:
        return n_other_pred + 1
    if other_et == "over instance" and n_other_pred > 1:
        return n_other_pred - 1

    inst = meta.get("instance_type", "")
    if inst == "1C1I":
        return 1
    return None


def classify_by_count_and_oks(n_pred, n_gt, oks):
    """Determine error_type from instance counts and OKS."""
    if n_gt is not None:
        if n_pred > n_gt:
            return "over instance"
        if n_pred < n_gt:
            return "less instance"

    if oks is None:
        return ""
    if oks >= 0.85:
        return "good"
    if oks >= 0.5:
        return "jitter"
    return "miss"


def main():
    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries.\n")

    backup_path = INPUT_JSON.with_suffix(f".pre_yolo8n_fix_{datetime.now():%Y%m%d_%H%M%S}.json")
    shutil.copy2(INPUT_JSON, backup_path)
    print(f"Backup: {backup_path}\n")

    fix_log = []
    stats = Counter()

    for entry in data:
        meta = entry["metadata"]
        dataset = meta.get("dataset", "")
        sample_id = meta.get("sample_id", "")

        for side in ("a", "b"):
            if meta.get(f"model_{side}_id") != "yolo8n":
                continue
            pp = meta.get(f"error_type_per_person_{side}", [])
            if len(pp) > 0:
                continue

            preds = load_yolo8n_predictions(dataset, sample_id)
            if preds is None:
                stats["pred_file_missing"] += 1
                continue

            n_pred = len(preds)
            n_gt = infer_n_gt(meta, side)
            oks = meta.get(f"score_{side}")
            old_et = meta.get(f"error_type_{side}", "")

            new_et = classify_by_count_and_oks(n_pred, n_gt, oks)

            per_person = []
            n_matched = min(n_pred, n_gt) if n_gt is not None else n_pred
            for _ in range(n_matched):
                if oks and oks >= 0.85:
                    per_person.append("good")
                elif oks and oks >= 0.5:
                    per_person.append("jitter")
                else:
                    per_person.append("miss")
            n_extra = n_pred - n_matched
            for _ in range(n_extra):
                per_person.append("miss")

            meta[f"error_type_{side}"] = new_et
            meta[f"error_type_per_person_{side}"] = per_person
            meta[f"_yolo8n_n_pred_{side}"] = n_pred
            meta[f"_yolo8n_n_gt_inferred_{side}"] = n_gt

            bpk = f"{sample_id}|{dataset}|{meta['model_a_id']}|{meta['model_b_id']}"
            changed = old_et != new_et
            if changed:
                stats["changed"] += 1
                fix_log.append(
                    f"[{bpk}] enc={meta['encoding']}: error_type_{side} '{old_et}' → '{new_et}' "
                    f"(n_pred={n_pred}, n_gt={n_gt}, oks={oks})"
                )
            else:
                stats["unchanged"] += 1
            stats[f"new_type_{new_et}"] += 1

    inst_fixes = 0
    for entry in data:
        meta = entry["metadata"]
        pp_a = meta.get("error_type_per_person_a", [])
        pp_b = meta.get("error_type_per_person_b", [])
        if meta.get("instance_type") == "1C1I" and (len(pp_a) > 1 or len(pp_b) > 1):
            meta["instance_type"] = "1CnI"
            inst_fixes += 1

    syn_mismatches = 0
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
            candidates = [et_a, et_b]
            non_good = [t for t in candidates if t and t != "good"]
            new_syn = non_good[0] if non_good else ""
            old_syn = meta["synthetic_error_type"]
            meta["synthetic_error_type"] = new_syn
            if not new_syn:
                meta["synthetic"] = False
            syn_mismatches += 1

    print("Recomputation stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  instance_type fixes: {inst_fixes}")
    print(f"  synthetic_error_type synced: {syn_mismatches}")
    print()

    with open(INPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {INPUT_JSON}")

    log_path = SCRIPT_DIR / "recompute_yolo8n_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Recomputed at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Backup: {backup_path}\n\n")
        for k, v in sorted(stats.items()):
            f.write(f"{k}: {v}\n")
        f.write(f"synthetic_error_type synced: {syn_mismatches}\n")
        f.write(f"\nDetailed changes:\n\n")
        for line in fix_log:
            f.write(line + "\n")
    print(f"Log: {log_path}")
    print("\nDone. Run validate_pairwise.py to verify.")


if __name__ == "__main__":
    main()
