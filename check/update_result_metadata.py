#!/usr/bin/env python3
"""
Update metadata in work_dirs_judge_keypoint/*/result.json
using the fixed pairwise_encoded_480.json.

This allows re-running analyze_judge_results_by_source.py
with corrected error_type, instance_type, and synthetic labels.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIXED_JSON = REPO_ROOT / "pairwise_encoded_480.json"
WORK_DIR = REPO_ROOT / "work_dirs_judge_keypoint"

METADATA_FIELDS = [
    "error_type_a", "error_type_b",
    "error_type_per_person_a", "error_type_per_person_b",
    "instance_type",
    "synthetic", "synthetic_error_type",
]


def build_lookup(fixed_data):
    """Build lookup: (sample_id, dataset, model_a_id, model_b_id, encoding) -> metadata"""
    lookup = {}
    for entry in fixed_data:
        m = entry["metadata"]
        key = (m["sample_id"], m["dataset"], m["model_a_id"], m["model_b_id"], m["encoding"])
        lookup[key] = m
    return lookup


def main():
    print(f"Loading fixed data: {FIXED_JSON}")
    with open(FIXED_JSON) as f:
        fixed_data = json.load(f)
    lookup = build_lookup(fixed_data)
    print(f"Built lookup with {len(lookup)} entries.\n")

    model_dirs = sorted([
        d for d in WORK_DIR.iterdir()
        if d.is_dir() and d.name not in ("analysis_by_source", "work_dirs_judge_keypoint")
    ])
    print(f"Found {len(model_dirs)} model result directories.\n")

    total_updated = 0
    total_not_found = 0

    for model_dir in model_dirs:
        result_path = model_dir / "result.json"
        if not result_path.exists():
            continue

        with open(result_path) as f:
            results = json.load(f)

        updated = 0
        not_found = 0
        for item in results:
            m = item.get("metadata", {})
            key = (m.get("sample_id"), m.get("dataset"),
                   m.get("model_a_id"), m.get("model_b_id"), m.get("encoding"))
            fixed_meta = lookup.get(key)
            if fixed_meta is None:
                not_found += 1
                continue

            changed = False
            for field in METADATA_FIELDS:
                old = m.get(field)
                new = fixed_meta.get(field)
                if old != new:
                    m[field] = new
                    changed = True
            if changed:
                updated += 1

        backup_path = model_dir / "result.json.bak"
        if not backup_path.exists():
            shutil.copy2(result_path, backup_path)

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  {model_dir.name}: {updated} updated, {not_found} not found")
        total_updated += updated
        total_not_found += not_found

    print(f"\nTotal: {total_updated} entries updated, {total_not_found} not found.")
    print("Backups saved as result.json.bak in each model directory.")
    print("\nNow run: python work_dirs_judge_keypoint/analyze_judge_results_by_source.py")


if __name__ == "__main__":
    main()
