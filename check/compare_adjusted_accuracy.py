#!/usr/bin/env python3
"""
Compare MLLM judge accuracy using original OKS-based answer vs adjusted_score-based answer.

Loads result.json files and the fixed pairwise_encoded_480.json,
then evaluates accuracy under both scoring schemes.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIXED_JSON = REPO_ROOT / "pairwise_encoded_480.json"
WORK_DIR = REPO_ROOT / "work_dirs_judge_keypoint"


def extract_answer(text):
    if not text:
        return None
    while isinstance(text, list):
        text = text[0] if text else None
    if not isinstance(text, str):
        return None
    m = re.match(r'^\s*([AB])(?:\.|:|\s|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([AB])\b', text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_lookup(fixed_data):
    lookup = {}
    for entry in fixed_data:
        m = entry["metadata"]
        key = (m["sample_id"], m["dataset"], m["model_a_id"], m["model_b_id"], m["encoding"])
        lookup[key] = m
    return lookup


def main():
    with open(FIXED_JSON) as f:
        fixed = json.load(f)
    lookup = build_lookup(fixed)

    model_dirs = sorted([
        d for d in WORK_DIR.iterdir()
        if d.is_dir() and d.name not in ("analysis_by_source", "work_dirs_judge_keypoint")
    ])

    original_by_model = defaultdict(lambda: {"correct": 0, "total": 0})
    adjusted_by_model = defaultdict(lambda: {"correct": 0, "total": 0})
    original_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    adjusted_by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for model_dir in model_dirs:
        bak = model_dir / "result.json.bak"
        main_f = model_dir / "result.json"
        result_path = bak if bak.exists() else main_f
        if not result_path.exists():
            continue

        with open(result_path) as f:
            results = json.load(f)

        model_name = model_dir.name
        for item in results:
            m = item.get("metadata", {})
            key = (m.get("sample_id"), m.get("dataset"),
                   m.get("model_a_id"), m.get("model_b_id"), m.get("encoding"))
            fixed_meta = lookup.get(key)
            if not fixed_meta:
                continue

            response = extract_answer(item.get("response"))
            if response is None:
                continue

            original_answer = extract_answer(item.get("answer"))
            adjusted_answer = fixed_meta.get("adjusted_answer")

            syn_type = fixed_meta.get("synthetic_error_type") or "model_based"

            if original_answer:
                original_by_model[model_name]["total"] += 1
                if response == original_answer:
                    original_by_model[model_name]["correct"] += 1
                original_by_type[syn_type]["total"] += 1
                if response == original_answer:
                    original_by_type[syn_type]["correct"] += 1

            if adjusted_answer:
                adjusted_by_model[model_name]["total"] += 1
                if response == adjusted_answer:
                    adjusted_by_model[model_name]["correct"] += 1
                adjusted_by_type[syn_type]["total"] += 1
                if response == adjusted_answer:
                    adjusted_by_type[syn_type]["correct"] += 1

    print("=" * 80)
    print("ACCURACY COMPARISON: Original OKS Answer vs Adjusted Score Answer")
    print("=" * 80)

    print(f"\n{'Model':<50s} {'OKS':>8s} {'Adj':>8s} {'Delta':>8s}")
    print("-" * 80)
    o_total_c, o_total_t, a_total_c, a_total_t = 0, 0, 0, 0
    for model in sorted(original_by_model.keys()):
        o = original_by_model[model]
        a = adjusted_by_model[model]
        o_acc = o["correct"] / o["total"] * 100 if o["total"] else 0
        a_acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        delta = a_acc - o_acc
        print(f"{model:<50s} {o_acc:7.1f}% {a_acc:7.1f}% {delta:+7.1f}%")
        o_total_c += o["correct"]; o_total_t += o["total"]
        a_total_c += a["correct"]; a_total_t += a["total"]

    o_avg = o_total_c / o_total_t * 100 if o_total_t else 0
    a_avg = a_total_c / a_total_t * 100 if a_total_t else 0
    print("-" * 80)
    print(f"{'OVERALL':<50s} {o_avg:7.1f}% {a_avg:7.1f}% {a_avg - o_avg:+7.1f}%")

    print(f"\n\n{'Error Type':<20s} {'OKS':>8s} {'N':>6s} {'Adj':>8s} {'N':>6s} {'Delta':>8s}")
    print("-" * 60)
    for et in sorted(set(list(original_by_type.keys()) + list(adjusted_by_type.keys()))):
        o = original_by_type[et]
        a = adjusted_by_type[et]
        o_acc = o["correct"] / o["total"] * 100 if o["total"] else 0
        a_acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        delta = a_acc - o_acc
        marker = " ***" if abs(delta) > 3 else ""
        print(f"{et:<20s} {o_acc:7.1f}% {o['total']:>6d} {a_acc:7.1f}% {a['total']:>6d} {delta:+7.1f}%{marker}")


if __name__ == "__main__":
    main()
