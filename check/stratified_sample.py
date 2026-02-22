#!/usr/bin/env python3
"""
Stratified sampling from pairwise_encoded_480.json.

Excludes flagged base pairs (from check/flagged_entries.json), then performs
stratified sampling to produce a balanced subset of base pairs with all 4
encoding variants.

Strategy:
  1. Keep ALL synthetic base pairs (already balanced by error type, precious).
  2. Stratify-sample random base pairs by (dataset × instance_type) to fill
     remaining quota.

Designed for iterative use:
  - Run check/validate_pairwise.py  -> produces flagged_entries.json
  - Run check/stratified_sample.py  -> produces sampled output
  - Manually fix flagged entries
  - Re-run validate -> re-run sample

Usage:
  python check/stratified_sample.py [--target N] [--seed S] [--output PATH]

  --target   Target number of base pairs (default: 300)
  --seed     Random seed (default: 42)
  --output   Output JSON path (default: check/sampled_{target}.json)
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_JSON = REPO_ROOT / "pairwise_encoded_480.json"
FLAGGED_JSON = SCRIPT_DIR / "flagged_entries.json"


def base_pair_key(meta):
    return (
        meta.get("sample_id", ""),
        meta.get("dataset", ""),
        meta.get("model_a_id", ""),
        meta.get("model_b_id", ""),
    )


def bpk_str(meta):
    k = base_pair_key(meta)
    return f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"


def stratified_sample(items, n, key_fn, rng):
    """
    Sample n items from items, stratified by key_fn.
    Each stratum gets floor(n * stratum_size / total) items, with remainders
    distributed round-robin to the smallest strata first (to boost minority groups).
    """
    strata = defaultdict(list)
    for item in items:
        strata[key_fn(item)].append(item)

    total = len(items)
    if n >= total:
        return list(items)

    allocation = {}
    remainder_pool = []
    base_allocated = 0

    for key, group in strata.items():
        share = n * len(group) / total
        floor_share = int(math.floor(share))
        allocation[key] = floor_share
        base_allocated += floor_share
        remainder_pool.append((share - floor_share, key))

    leftover = n - base_allocated
    remainder_pool.sort(key=lambda x: (-x[0], len(strata[x[1]])))
    for i in range(min(leftover, len(remainder_pool))):
        allocation[remainder_pool[i][1]] += 1

    result = []
    for key, group in strata.items():
        k = min(allocation.get(key, 0), len(group))
        result.extend(rng.sample(group, k))

    return result


def main():
    parser = argparse.ArgumentParser(description="Stratified sampling of OK base pairs")
    parser.add_argument("--target", type=int, default=300,
                        help="Target number of base pairs (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: check/sampled_{target}.json)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output) if args.output else SCRIPT_DIR / f"sampled_{args.target}.json"

    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    flagged_bpks = set()
    if FLAGGED_JSON.exists():
        with open(FLAGGED_JSON, "r") as f:
            flagged = json.load(f)
        flagged_bpks = {r["base_pair_key"] for r in flagged}
        print(f"Loaded {len(flagged_bpks)} flagged base pairs from {FLAGGED_JSON}")
    else:
        print(f"No flagged entries file found at {FLAGGED_JSON}, using all base pairs.")

    groups = defaultdict(list)
    for entry in data:
        meta = entry["metadata"]
        key = base_pair_key(meta)
        groups[key].append(entry)

    ok_groups = {k: v for k, v in groups.items() if bpk_str(v[0]["metadata"]) not in flagged_bpks}
    print(f"Total base pairs: {len(groups)}")
    print(f"OK base pairs (after excluding flagged): {len(ok_groups)}")
    print()

    representatives = {}
    for key, entries in ok_groups.items():
        enc_a = [e for e in entries if e["metadata"]["encoding"] == "a"]
        representatives[key] = enc_a[0] if enc_a else entries[0]

    synthetic_keys = [k for k, e in representatives.items() if e["metadata"].get("synthetic")]
    random_keys = [k for k, e in representatives.items() if not e["metadata"].get("synthetic")]

    print(f"OK synthetic base pairs: {len(synthetic_keys)}")
    print(f"OK random base pairs:    {len(random_keys)}")

    selected_keys = list(synthetic_keys)
    remaining_quota = args.target - len(selected_keys)

    if remaining_quota <= 0:
        print(f"\nSynthetic alone ({len(synthetic_keys)}) meets target ({args.target}).")
        print(f"Keeping all synthetic, no random sampling needed.")
        remaining_quota = 0
    elif remaining_quota >= len(random_keys):
        print(f"\nNeed {remaining_quota} random, but only {len(random_keys)} available. Using all.")
        selected_keys.extend(random_keys)
        remaining_quota = 0
    else:
        print(f"\nSampling {remaining_quota} from {len(random_keys)} random base pairs (stratified)...")

        def stratum_key(k):
            e = representatives[k]
            m = e["metadata"]
            return (m.get("dataset", ""), m.get("instance_type", ""))

        sampled_random = stratified_sample(random_keys, remaining_quota, stratum_key, rng)
        selected_keys.extend(sampled_random)

    selected_set = set(selected_keys)
    output_entries = []
    for key in selected_keys:
        output_entries.extend(ok_groups[key])

    print(f"\n{'='*60}")
    print(f"SAMPLING RESULT")
    print(f"{'='*60}")
    print(f"Selected base pairs: {len(selected_set)}")
    print(f"Selected entries (all encodings): {len(output_entries)}")
    print()

    sel_reps = [representatives[k] for k in selected_keys]

    print("Distribution of selected base pairs:")
    print()

    syn_dist = Counter(e["metadata"].get("synthetic", False) for e in sel_reps)
    print(f"  synthetic:  True={syn_dist.get(True, 0)}, False={syn_dist.get(False, 0)}")

    syn_type_dist = Counter(
        e["metadata"].get("synthetic_error_type", "N/A")
        for e in sel_reps if e["metadata"].get("synthetic")
    )
    print(f"  synthetic_error_type: {dict(syn_type_dist)}")

    ds_dist = Counter(e["metadata"].get("dataset", "") for e in sel_reps)
    print(f"  dataset: {dict(ds_dist)}")

    inst_dist = Counter(e["metadata"].get("instance_type", "") for e in sel_reps)
    print(f"  instance_type: {dict(inst_dist)}")

    eta_dist = Counter(e["metadata"].get("error_type_a", "") for e in sel_reps)
    print(f"  error_type_a: {dict(eta_dist)}")
    print()

    all_reps = [representatives[k] for k in ok_groups]
    print("Comparison with full OK pool:")
    print()
    print(f"  {'Category':<25} {'OK Pool':>10} {'Selected':>10} {'Ratio':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

    for label, sel_counter, all_items, key_fn in [
        ("dataset", ds_dist, all_reps, lambda e: e["metadata"].get("dataset", "")),
        ("instance_type", inst_dist, all_reps, lambda e: e["metadata"].get("instance_type", "")),
    ]:
        all_counter = Counter(key_fn(e) for e in all_items)
        for k in sorted(set(list(sel_counter.keys()) + list(all_counter.keys()))):
            s = sel_counter.get(k, 0)
            a = all_counter.get(k, 0)
            ratio = f"{s/a*100:.0f}%" if a > 0 else "N/A"
            print(f"  {label}={k:<14} {a:>10} {s:>10} {ratio:>8}")
    print()

    with open(output_path, "w") as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)
    print(f"Output written to {output_path}")
    print(f"  {len(selected_set)} base pairs x 4 encodings = {len(output_entries)} entries")

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "target": args.target,
        "seed": args.seed,
        "total_base_pairs": len(groups),
        "ok_base_pairs": len(ok_groups),
        "flagged_base_pairs": len(flagged_bpks),
        "selected_base_pairs": len(selected_set),
        "selected_entries": len(output_entries),
        "selected_keys": [f"{k[0]}|{k[1]}|{k[2]}|{k[3]}" for k in selected_keys],
        "distribution": {
            "synthetic": dict(syn_dist),
            "synthetic_error_type": dict(syn_type_dist),
            "dataset": dict(ds_dist),
            "instance_type": dict(inst_dist),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {manifest_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
