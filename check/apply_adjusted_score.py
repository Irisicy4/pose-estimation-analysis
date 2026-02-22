#!/usr/bin/env python3
"""
Apply instance-aware adjusted_score to pairwise_encoded_480.json.

For each entry, computes:
  - n_pred_{a,b}: number of predicted instances (from per_person length)
  - n_gt: inferred from the data
  - instance_recall_{a,b}: min(n_pred, n_gt) / n_gt
  - adjusted_score_{a,b}: score * instance_recall (penalizes missing instances)
  - adjusted_score_difference: |adjusted_score_a - adjusted_score_b|
  - adjusted_answer: which model is better based on adjusted_score

The adjusted_answer may differ from the original answer when one model has
high OKS but missed many instances (less instance).
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_JSON = REPO_ROOT / "pairwise_encoded_480.json"


def infer_n_gt_and_n_pred(meta):
    """Infer n_gt and n_pred from available metadata."""
    pp_a = meta.get("error_type_per_person_a", [])
    pp_b = meta.get("error_type_per_person_b", [])
    et_a = meta.get("error_type_a", "")
    et_b = meta.get("error_type_b", "")
    n_pred_a = meta.get("_yolo8n_n_pred_a") or len(pp_a)
    n_pred_b = meta.get("_yolo8n_n_pred_b") or len(pp_b)

    n_gt_explicit = meta.get("_yolo8n_n_gt_a") or meta.get("_yolo8n_n_gt_b")
    if n_gt_explicit is not None:
        return n_pred_a, n_pred_b, n_gt_explicit

    n_gt_candidates = []

    for side_pp, side_et, side_np in [
        (pp_a, et_a, n_pred_a), (pp_b, et_b, n_pred_b)
    ]:
        if side_et in ("good", "jitter", "miss", "inversion", "swap") and len(side_pp) > 0:
            n_gt_candidates.append(len(side_pp))
        elif side_et == "less instance":
            n_gt_candidates.append(len(side_pp) + 1)
        elif side_et == "over instance" and len(side_pp) > 1:
            n_gt_candidates.append(len(side_pp) - 1)

    if n_gt_candidates:
        n_gt = max(n_gt_candidates)
    else:
        inst = meta.get("instance_type", "")
        n_gt = 1 if inst == "1C1I" else max(n_pred_a, n_pred_b, 1)

    return n_pred_a, n_pred_b, n_gt


def compute_instance_recall(n_pred, n_gt):
    if n_gt <= 0:
        return 1.0
    return min(n_pred, n_gt) / n_gt


def main():
    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries.\n")

    backup_path = INPUT_JSON.with_suffix(
        f".pre_adjusted_score_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(INPUT_JSON, backup_path)
    print(f"Backup: {backup_path}\n")

    answer_changes = 0
    stats = Counter()

    for entry in data:
        meta = entry["metadata"]
        score_a = meta.get("score_a")
        score_b = meta.get("score_b")

        n_pred_a, n_pred_b, n_gt = infer_n_gt_and_n_pred(meta)

        recall_a = compute_instance_recall(n_pred_a, n_gt)
        recall_b = compute_instance_recall(n_pred_b, n_gt)

        meta["n_pred_a"] = n_pred_a
        meta["n_pred_b"] = n_pred_b
        meta["n_gt"] = n_gt
        meta["instance_recall_a"] = round(recall_a, 6)
        meta["instance_recall_b"] = round(recall_b, 6)

        if score_a is not None and score_b is not None:
            adj_a = round(score_a * recall_a, 6)
            adj_b = round(score_b * recall_b, 6)
            meta["adjusted_score_a"] = adj_a
            meta["adjusted_score_b"] = adj_b
            meta["adjusted_score_difference"] = round(abs(adj_a - adj_b), 6)

            old_answer = meta.get("answer") or entry.get("answer")
            if adj_a > adj_b:
                new_answer = "A"
            elif adj_b > adj_a:
                new_answer = "B"
            else:
                new_answer = old_answer

            if old_answer and new_answer != old_answer:
                answer_changes += 1
                stats[f"flip_{old_answer}->{new_answer}"] += 1

            meta["adjusted_answer"] = new_answer

        if recall_a < 1.0 or recall_b < 1.0:
            stats["has_instance_penalty"] += 1
        else:
            stats["no_penalty"] += 1

    with open(INPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Results:")
    print(f"  Entries with instance penalty: {stats['has_instance_penalty']}")
    print(f"  Entries without penalty:       {stats['no_penalty']}")
    print(f"  Answer flips (adjusted vs original): {answer_changes}")
    for k, v in sorted(stats.items()):
        if k.startswith("flip_"):
            print(f"    {k}: {v}")
    print(f"\nUpdated {INPUT_JSON}")

    # Show examples of answer flips
    flips = []
    for entry in data:
        meta = entry["metadata"]
        if meta.get("encoding") != "a":
            continue
        old = meta.get("answer") or entry.get("answer")
        new = meta.get("adjusted_answer")
        if old and new and old != new:
            flips.append({
                "sample_id": meta["sample_id"],
                "dataset": meta["dataset"],
                "models": f"{meta['model_a_id']} vs {meta['model_b_id']}",
                "score_a": meta.get("score_a"),
                "score_b": meta.get("score_b"),
                "adj_a": meta.get("adjusted_score_a"),
                "adj_b": meta.get("adjusted_score_b"),
                "recall_a": meta.get("instance_recall_a"),
                "recall_b": meta.get("instance_recall_b"),
                "n_pred_a": meta.get("n_pred_a"),
                "n_pred_b": meta.get("n_pred_b"),
                "n_gt": meta.get("n_gt"),
                "old_answer": old,
                "new_answer": new,
            })

    if flips:
        print(f"\nAnswer flip examples (base pairs, first 10):")
        for f in flips[:10]:
            print(f"  {f['sample_id']}|{f['dataset']}|{f['models']}")
            print(f"    OKS: A={f['score_a']:.4f} B={f['score_b']:.4f} → answer={f['old_answer']}")
            print(f"    recall: A={f['recall_a']:.2f} (n_pred={f['n_pred_a']}) B={f['recall_b']:.2f} (n_pred={f['n_pred_b']}) n_gt={f['n_gt']}")
            print(f"    adjusted: A={f['adj_a']:.4f} B={f['adj_b']:.4f} → answer={f['new_answer']}")

    log_path = SCRIPT_DIR / "adjusted_score_log.txt"
    with open(log_path, "w") as lf:
        lf.write(f"Applied at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        lf.write(f"Answer flips: {answer_changes}\n")
        lf.write(f"Entries with penalty: {stats['has_instance_penalty']}\n\n")
        for f in flips:
            lf.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
