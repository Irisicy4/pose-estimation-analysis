#!/usr/bin/env python3
"""
Validate pairwise_encoded_480.json — automated checks for data quality.

Checks performed:
  1. Encoding completeness: each base pair has exactly 4 encodings (a,b,c,d)
  2. Answer vs score direction consistency
  3. Cross-encoding field consistency within each base pair
  4. Encoding-specific format (media length, prompt content, media path)
  5. Media file path correctness and existence
  6. Error type heuristic cross-validation
  7. Instance type reasonability
  8. Basic field completeness and value range

Outputs:
  - check/report.json          per-entry detailed results
  - check/summary.txt          human-readable summary
  - check/flagged_entries.json  entries with at least one failure
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_JSON = REPO_ROOT / "pairwise_encoded_480.json"

VALID_ENCODINGS = {"a", "b", "c", "d"}
VALID_ERROR_TYPES = {"good", "jitter", "miss", "inversion", "swap", "over instance", "less instance", ""}
ENCODING_MEDIA_DIR = {
    "a": "encode_same_color",
    "b": "encode_per_person_color",
    "c": "encode_per_body_part",
}


def base_pair_key(meta):
    return (
        meta.get("sample_id", ""),
        meta.get("dataset", ""),
        meta.get("model_a_id", ""),
        meta.get("model_b_id", ""),
    )


def check_basic_fields(entry, idx):
    """Check 8: basic field completeness and value range."""
    issues = []
    meta = entry.get("metadata")
    if meta is None:
        issues.append("missing metadata")
        return issues

    if entry.get("answer") not in ("A", "B"):
        issues.append(f"invalid answer: {entry.get('answer')!r}")
    if entry.get("choices") != ["A", "B"]:
        issues.append(f"invalid choices: {entry.get('choices')!r}")
    if not entry.get("prompt"):
        issues.append("empty prompt")
    if not entry.get("media"):
        issues.append("empty media")

    for field in ("sample_id", "dataset", "model_a_id", "model_b_id", "encoding"):
        if not meta.get(field):
            issues.append(f"missing metadata.{field}")

    score_a = meta.get("score_a")
    score_b = meta.get("score_b")
    if score_a is not None and not (0.0 <= score_a <= 1.0):
        issues.append(f"score_a out of range: {score_a}")
    if score_b is not None and not (0.0 <= score_b <= 1.0):
        issues.append(f"score_b out of range: {score_b}")

    enc = meta.get("encoding", "")
    if enc not in VALID_ENCODINGS:
        issues.append(f"invalid encoding: {enc!r}")

    for et_field in ("error_type_a", "error_type_b"):
        val = meta.get(et_field, "")
        if val not in VALID_ERROR_TYPES:
            issues.append(f"invalid {et_field}: {val!r}")

    if meta.get("synthetic") is True and not meta.get("synthetic_error_type"):
        issues.append("synthetic=true but missing synthetic_error_type")

    return issues


def check_answer_score_consistency(entry):
    """Check 2: answer direction should match score direction."""
    issues = []
    meta = entry.get("metadata", {})
    score_a = meta.get("score_a")
    score_b = meta.get("score_b")
    answer = entry.get("answer")

    if score_a is None or score_b is None or answer not in ("A", "B"):
        return issues

    if answer == "A" and score_a < score_b:
        diff = score_b - score_a
        issues.append(f"answer=A but score_a({score_a:.4f}) < score_b({score_b:.4f}), diff={diff:.4f}")
    elif answer == "B" and score_b < score_a:
        diff = score_a - score_b
        issues.append(f"answer=B but score_b({score_b:.4f}) < score_a({score_a:.4f}), diff={diff:.4f}")

    return issues


def check_encoding_format(entry):
    """Check 4: encoding-specific format validation."""
    issues = []
    meta = entry.get("metadata", {})
    enc = meta.get("encoding", "")
    media = entry.get("media", [])
    prompt = entry.get("prompt", "")

    if enc in ("a", "b", "c"):
        if len(media) != 3:
            issues.append(f"encoding={enc} should have 3 media items, got {len(media)}")
        if len(media) >= 3:
            expected_dir = ENCODING_MEDIA_DIR.get(enc, "")
            if expected_dir not in media[1]:
                issues.append(f"encoding={enc} media[1] path missing '{expected_dir}': {media[1]}")
            if expected_dir not in media[2]:
                issues.append(f"encoding={enc} media[2] path missing '{expected_dir}': {media[2]}")
        if "<image>" not in prompt:
            issues.append(f"encoding={enc} prompt missing <image> placeholder")

    elif enc == "d":
        if len(media) != 1:
            issues.append(f"encoding=d should have 1 media item (original only), got {len(media)}")
        has_json_content = any(kw in prompt for kw in ("person_id", "keypoints", "name"))
        if not has_json_content:
            issues.append("encoding=d prompt missing JSON keypoint content")
        if prompt.count("<image>") != 1:
            issues.append(f"encoding=d prompt should have exactly 1 <image>, got {prompt.count('<image>')}")

    return issues


def check_media_paths(entry):
    """Check 5: media file path format and naming convention (not file existence)."""
    issues = []
    meta = entry.get("metadata", {})
    enc = meta.get("encoding", "")
    media = entry.get("media", [])
    model_a_id = meta.get("model_a_id", "")
    model_b_id = meta.get("model_b_id", "")

    if enc in ("a", "b", "c") and len(media) >= 3:
        for i, label in [(1, "A"), (2, "B")]:
            path_str = media[i]
            if f"_{label}.jpg" not in path_str:
                issues.append(f"media[{i}] should end with _{label}.jpg: {path_str}")

        if model_a_id and model_a_id not in media[1] and model_b_id not in media[1]:
            issues.append(f"media[1] path doesn't contain model ids: {media[1]}")
        if model_b_id and model_a_id not in media[2] and model_b_id not in media[2]:
            issues.append(f"media[2] path doesn't contain model ids: {media[2]}")

    return issues


def check_media_file_existence(entry):
    """Check 5b: media file existence on disk (optional, may fail if data is remote)."""
    issues = []
    media = entry.get("media", [])
    for i, path_str in enumerate(media):
        full_path = REPO_ROOT / path_str
        if not full_path.exists():
            issues.append(f"media[{i}] file not found: {path_str}")
    return issues


def check_error_type_heuristics(entry):
    """Check 6: error type heuristic cross-validation."""
    issues = []
    meta = entry.get("metadata", {})
    score_a = meta.get("score_a")
    score_b = meta.get("score_b")
    et_a = meta.get("error_type_a", "")
    et_b = meta.get("error_type_b", "")
    syn = meta.get("synthetic", False)
    syn_type = meta.get("synthetic_error_type", "")
    pp_a = meta.get("error_type_per_person_a", [])
    pp_b = meta.get("error_type_per_person_b", [])

    if syn and syn_type:
        match = (et_a == syn_type or et_b == syn_type or
                 (et_a == "good" and et_b == syn_type) or
                 (et_b == "good" and et_a == syn_type))
        if not match:
            issues.append(
                f"synthetic_error_type={syn_type!r} but "
                f"error_type_a={et_a!r}, error_type_b={et_b!r}"
            )

    # --- jitter checks ---
    if et_a == "jitter" and score_a is not None and score_a < 0.5:
        issues.append(f"error_type_a=jitter but score_a={score_a:.4f} (expected > 0.5)")
    if et_b == "jitter" and score_b is not None and score_b < 0.5:
        issues.append(f"error_type_b=jitter but score_b={score_b:.4f} (expected > 0.5)")

    if et_a == "jitter" and et_b == "jitter" and score_a is not None and score_b is not None:
        diff = abs(score_a - score_b)
        if diff > 0.3:
            issues.append(f"both jitter but score_difference={diff:.4f} (expected < 0.3)")

    # --- over instance checks ---
    if et_a == "over instance" and len(pp_a) <= 1:
        issues.append(f"error_type_a=over instance but per_person_a has {len(pp_a)} entries")
    if et_b == "over instance" and len(pp_b) <= 1:
        issues.append(f"error_type_b=over instance but per_person_b has {len(pp_b)} entries")

    # --- miss + high score checks ---
    if et_a == "miss" and score_a is not None and score_a > 0.85:
        issues.append(f"error_type_a=miss but score_a={score_a:.4f} (unexpectedly high)")
    if et_b == "miss" and score_b is not None and score_b > 0.85:
        issues.append(f"error_type_b=miss but score_b={score_b:.4f} (unexpectedly high)")

    # --- less instance: per_person empty but score high (prediction load failure) ---
    if et_a == "less instance" and len(pp_a) == 0 and score_a is not None and score_a > 0.5:
        issues.append(
            f"error_type_a=less instance with per_person_a=[] but score_a={score_a:.4f} "
            f"(likely prediction load failure — score suggests detections exist)"
        )
    if et_b == "less instance" and len(pp_b) == 0 and score_b is not None and score_b > 0.5:
        issues.append(
            f"error_type_b=less instance with per_person_b=[] but score_b={score_b:.4f} "
            f"(likely prediction load failure — score suggests detections exist)"
        )

    # --- swap: both sides swap with tiny score diff (ambiguous, unjudgeable) ---
    if et_a == "swap" and et_b == "swap" and score_a is not None and score_b is not None:
        diff = abs(score_a - score_b)
        if diff < 0.05:
            issues.append(
                f"both sides swap with score_diff={diff:.4f} "
                f"(< 0.05, visually indistinguishable for MLLM)"
            )

    return issues


def check_instance_type(entry):
    """Check 7: instance_type reasonability."""
    issues = []
    meta = entry.get("metadata", {})
    inst = meta.get("instance_type", "")
    pp_a = meta.get("error_type_per_person_a", [])
    pp_b = meta.get("error_type_per_person_b", [])

    if inst not in ("1C1I", "1CnI"):
        issues.append(f"invalid instance_type: {inst!r}")
        return issues

    if inst == "1C1I":
        if len(pp_a) > 1:
            issues.append(f"instance_type=1C1I but error_type_per_person_a has {len(pp_a)} entries")
        if len(pp_b) > 1:
            issues.append(f"instance_type=1C1I but error_type_per_person_b has {len(pp_b)} entries")

    return issues


def check_cross_encoding_consistency(groups):
    """Check 1 & 3: encoding completeness and cross-encoding field consistency."""
    results = {}
    CONSISTENT_FIELDS = [
        "answer", "score_a", "score_b", "model_a_id", "model_b_id",
        "error_type_a", "error_type_b", "instance_type", "sample_id",
        "dataset", "synthetic", "synthetic_error_type",
    ]

    for key, entries in groups.items():
        issues = []
        encodings_found = {e["metadata"]["encoding"] for e in entries}

        missing = VALID_ENCODINGS - encodings_found
        if missing:
            issues.append(f"missing encodings: {sorted(missing)}")
        extra = encodings_found - VALID_ENCODINGS
        if extra:
            issues.append(f"unexpected encodings: {sorted(extra)}")
        if len(entries) != 4:
            issues.append(f"expected 4 entries, got {len(entries)}")

        for field in CONSISTENT_FIELDS:
            values = set()
            for e in entries:
                if field == "answer":
                    values.add(e.get("answer"))
                else:
                    v = e["metadata"].get(field)
                    if isinstance(v, list):
                        v = tuple(v)
                    values.add(v)
            if len(values) > 1:
                issues.append(f"inconsistent {field} across encodings: {values}")

        results[key] = issues
    return results


def main():
    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries.\n")

    groups = defaultdict(list)
    for entry in data:
        meta = entry.get("metadata", {})
        key = base_pair_key(meta)
        groups[key].append(entry)

    print(f"Found {len(groups)} base pairs.\n")

    cross_enc_results = check_cross_encoding_consistency(groups)

    # Probe whether media files exist locally (check first entry)
    probe_media = data[0].get("media", [None])[0] if data else None
    media_exists_locally = probe_media and (REPO_ROOT / probe_media).exists()
    if not media_exists_locally:
        print("NOTE: Media files not found locally (data likely on remote server).")
        print("      Skipping file-existence checks; only checking path format.\n")

    all_results = []
    summary_counters = {
        "basic_fields": 0,
        "answer_score_consistency": 0,
        "encoding_format": 0,
        "media_path_format": 0,
        "error_type_heuristics": 0,
        "instance_type": 0,
        "cross_encoding": 0,
    }
    if media_exists_locally:
        summary_counters["media_file_existence"] = 0
    total_flagged = 0

    for idx, entry in enumerate(data):
        meta = entry.get("metadata", {})
        key = base_pair_key(meta)
        result = {
            "id": entry.get("id", str(idx)),
            "base_pair_key": f"{key[0]}|{key[1]}|{key[2]}|{key[3]}",
            "encoding": meta.get("encoding", "?"),
            "checks": {},
            "status": "ok",
        }

        checks = {
            "basic_fields": check_basic_fields(entry, idx),
            "answer_score_consistency": check_answer_score_consistency(entry),
            "encoding_format": check_encoding_format(entry),
            "media_path_format": check_media_paths(entry),
            "error_type_heuristics": check_error_type_heuristics(entry),
            "instance_type": check_instance_type(entry),
        }
        if media_exists_locally:
            checks["media_file_existence"] = check_media_file_existence(entry)

        cross_issues = cross_enc_results.get(key, [])
        checks["cross_encoding"] = cross_issues

        has_failure = False
        for check_name, issue_list in checks.items():
            if issue_list:
                has_failure = True
                summary_counters[check_name] += 1
            result["checks"][check_name] = issue_list if issue_list else "pass"

        if has_failure:
            result["status"] = "fail"
            total_flagged += 1

        all_results.append(result)

    flagged = [r for r in all_results if r["status"] == "fail"]

    flagged_base_pairs = set()
    for r in flagged:
        flagged_base_pairs.add(r["base_pair_key"])

    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total entries:        {len(data)}")
    print(f"Total base pairs:     {len(groups)}")
    print(f"Entries with issues:  {total_flagged} / {len(data)}")
    print(f"Base pairs with issues: {len(flagged_base_pairs)} / {len(groups)}")
    print()
    print("Failures by check category:")
    for check_name, count in sorted(summary_counters.items()):
        pct = count / len(data) * 100
        print(f"  {check_name:35s}  {count:5d}  ({pct:.1f}%)")
    print()

    answer_score_details = []
    for r in all_results:
        issues = r["checks"].get("answer_score_consistency", [])
        if isinstance(issues, list) and issues:
            answer_score_details.append((r["base_pair_key"], r["encoding"], issues))

    if answer_score_details:
        print("-" * 70)
        print("ANSWER vs SCORE DIRECTION MISMATCHES (sample):")
        print("-" * 70)
        shown = set()
        for bpk, enc, issues in answer_score_details:
            if bpk in shown:
                continue
            shown.add(bpk)
            if len(shown) > 20:
                print(f"  ... and {len(set(d[0] for d in answer_score_details)) - 20} more base pairs")
                break
            print(f"  [{bpk}] enc={enc}: {issues[0]}")
        print()

    cross_enc_issues = [(k, v) for k, v in cross_enc_results.items() if v]
    if cross_enc_issues:
        print("-" * 70)
        print(f"CROSS-ENCODING CONSISTENCY ISSUES ({len(cross_enc_issues)} base pairs):")
        print("-" * 70)
        for key, issues in cross_enc_issues[:20]:
            print(f"  [{key[0]}|{key[1]}|{key[2]}|{key[3]}]")
            for iss in issues:
                print(f"    - {iss}")
        if len(cross_enc_issues) > 20:
            print(f"  ... and {len(cross_enc_issues) - 20} more")
        print()

    et_heuristic_issues = []
    for r in all_results:
        issues = r["checks"].get("error_type_heuristics", [])
        if isinstance(issues, list) and issues:
            et_heuristic_issues.append((r["base_pair_key"], r["encoding"], issues))
    if et_heuristic_issues:
        print("-" * 70)
        print(f"ERROR TYPE HEURISTIC FLAGS ({len(et_heuristic_issues)} entries):")
        print("-" * 70)
        shown = set()
        for bpk, enc, issues in et_heuristic_issues:
            if bpk in shown:
                continue
            shown.add(bpk)
            if len(shown) > 20:
                print(f"  ... and {len(set(d[0] for d in et_heuristic_issues)) - 20} more base pairs")
                break
            for iss in issues:
                print(f"  [{bpk}] enc={enc}: {iss}")
        print()

    inst_issues = []
    for r in all_results:
        issues = r["checks"].get("instance_type", [])
        if isinstance(issues, list) and issues:
            inst_issues.append((r["base_pair_key"], r["encoding"], issues))
    if inst_issues:
        print("-" * 70)
        print(f"INSTANCE TYPE ISSUES ({len(inst_issues)} entries):")
        print("-" * 70)
        shown = set()
        for bpk, enc, issues in inst_issues:
            if bpk in shown:
                continue
            shown.add(bpk)
            if len(shown) > 15:
                print(f"  ... and {len(set(d[0] for d in inst_issues)) - 15} more")
                break
            for iss in issues:
                print(f"  [{bpk}]: {iss}")
        print()

    report_path = SCRIPT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full report written to {report_path}")

    flagged_path = SCRIPT_DIR / "flagged_entries.json"
    with open(flagged_path, "w") as f:
        json.dump(flagged, f, indent=2, ensure_ascii=False)
    print(f"Flagged entries written to {flagged_path}")

    summary_path = SCRIPT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total entries: {len(data)}\n")
        f.write(f"Total base pairs: {len(groups)}\n")
        f.write(f"Entries with issues: {total_flagged} / {len(data)}\n")
        f.write(f"Base pairs with issues: {len(flagged_base_pairs)} / {len(groups)}\n\n")
        f.write("Failures by check category:\n")
        for check_name, count in sorted(summary_counters.items()):
            pct = count / len(data) * 100
            f.write(f"  {check_name:35s}  {count:5d}  ({pct:.1f}%)\n")

        f.write(f"\n{'='*70}\n")
        f.write("BASE PAIRS BY STATUS:\n\n")
        ok_pairs = [k for k in groups if k not in
                    {tuple(r["base_pair_key"].split("|")) for r in flagged}]
        f.write(f"OK base pairs: {len(ok_pairs)}\n")
        f.write(f"Flagged base pairs: {len(flagged_base_pairs)}\n")

        f.write(f"\n{'='*70}\n")
        f.write("FLAGGED BASE PAIR DETAILS:\n\n")
        for bpk in sorted(flagged_base_pairs):
            f.write(f"--- {bpk} ---\n")
            for r in flagged:
                if r["base_pair_key"] == bpk:
                    for ck, cv in r["checks"].items():
                        if isinstance(cv, list) and cv:
                            for iss in cv:
                                f.write(f"  [{r['encoding']}] {ck}: {iss}\n")
            f.write("\n")

    print(f"Summary written to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
