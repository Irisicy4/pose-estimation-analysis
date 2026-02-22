#!/usr/bin/env python3
"""
Fix known data quality issues in pairwise_encoded_480.json.

Fixes applied:
  1. instance_type 1C1I → 1CnI when per_person has multiple entries
  2. over instance + mpii300 + per_person ≤ 1 → error_type set to ""
  3. miss + score > 0.85 → error_type changed to "jitter"
  4. yolo8n prediction load failure → error_type set to "" for yolo8n side
     (Root cause: 2rinse_error_type.py could not load yolo8n predictions,
      likely due to directory name mismatch on the server. All 1594 yolo8n
      entries in selected_pairwise_error_type.json have empty per_person.)

Creates a backup before modifying, and logs all changes.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_JSON = REPO_ROOT / "pairwise_encoded_480.json"


def base_pair_key_str(meta):
    return f"{meta.get('sample_id','')}|{meta.get('dataset','')}|{meta.get('model_a_id','')}|{meta.get('model_b_id','')}"


def main():
    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries.\n")

    backup_path = INPUT_JSON.with_suffix(f".pre_fix_{datetime.now():%Y%m%d_%H%M%S}.json")
    shutil.copy2(INPUT_JSON, backup_path)
    print(f"Backup saved to {backup_path}\n")

    fix_log = []
    fix_counts = {
        "instance_type": 0,
        "over_instance_mpii": 0,
        "miss_high_score": 0,
        "yolo8n_pred_failure": 0,
        "sync_synthetic_type": 0,
    }

    for entry in data:
        meta = entry.get("metadata", {})
        bpk = base_pair_key_str(meta)
        pp_a = meta.get("error_type_per_person_a", [])
        pp_b = meta.get("error_type_per_person_b", [])

        # Fix 1: instance_type 1C1I → 1CnI
        if meta.get("instance_type") == "1C1I" and (len(pp_a) > 1 or len(pp_b) > 1):
            old = meta["instance_type"]
            meta["instance_type"] = "1CnI"
            fix_log.append(f"[{bpk}] enc={meta.get('encoding')}: instance_type {old} → 1CnI (pp_a={len(pp_a)}, pp_b={len(pp_b)})")
            fix_counts["instance_type"] += 1

        # Fix 2: over instance + mpii300 + per_person ≤ 1 → error_type=""
        dataset = meta.get("dataset", "")
        for side in ("a", "b"):
            et_key = f"error_type_{side}"
            pp_key = f"error_type_per_person_{side}"
            et = meta.get(et_key, "")
            pp = meta.get(pp_key, [])
            if et == "over instance" and dataset == "mpii300" and len(pp) <= 1:
                meta[et_key] = ""
                fix_log.append(f"[{bpk}] enc={meta.get('encoding')}: {et_key} 'over instance' → '' (mpii300, per_person={len(pp)})")
                fix_counts["over_instance_mpii"] += 1

        # Fix 3: miss + score > 0.85 → jitter
        for side in ("a", "b"):
            et_key = f"error_type_{side}"
            score_key = f"score_{side}"
            et = meta.get(et_key, "")
            score = meta.get(score_key)
            if et == "miss" and score is not None and score > 0.85:
                meta[et_key] = "jitter"
                fix_log.append(f"[{bpk}] enc={meta.get('encoding')}: {et_key} 'miss' → 'jitter' (score={score:.4f})")
                fix_counts["miss_high_score"] += 1

        # Fix 4: yolo8n prediction load failure — per_person always empty
        for side in ("a", "b"):
            model_key = f"model_{side}_id"
            et_key = f"error_type_{side}"
            pp_key = f"error_type_per_person_{side}"
            if meta.get(model_key) == "yolo8n" and len(meta.get(pp_key, [])) == 0:
                old_et = meta.get(et_key, "")
                if old_et:
                    meta[et_key] = ""
                    fix_log.append(
                        f"[{bpk}] enc={meta.get('encoding')}: {et_key} '{old_et}' → '' "
                        f"(yolo8n prediction load failure, per_person empty)"
                    )
                    fix_counts["yolo8n_pred_failure"] += 1

        # Fix 5: sync synthetic_error_type after error_type corrections
        if meta.get("synthetic") and meta.get("synthetic_error_type"):
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
                fix_log.append(
                    f"[{bpk}] enc={meta.get('encoding')}: synthetic_error_type '{old_syn}' → '{new_syn}'"
                    f"{', synthetic → False' if not new_syn else ''}"
                    f" (sync after error_type correction, et_a='{et_a}', et_b='{et_b}')"
                )
                fix_counts["sync_synthetic_type"] += 1

    print("Fixes applied:")
    for fix_name, count in fix_counts.items():
        print(f"  {fix_name}: {count} entries")
    print(f"  total: {sum(fix_counts.values())} entries")
    print()

    with open(INPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {INPUT_JSON}")

    log_path = SCRIPT_DIR / "fix_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Fix applied at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Backup: {backup_path}\n\n")
        for fix_name, count in fix_counts.items():
            f.write(f"{fix_name}: {count} entries\n")
        f.write(f"\nDetailed changes:\n\n")
        for line in fix_log:
            f.write(line + "\n")
    print(f"Fix log written to {log_path}")
    print("\nDone. Run validate_pairwise.py to verify.")


if __name__ == "__main__":
    main()
