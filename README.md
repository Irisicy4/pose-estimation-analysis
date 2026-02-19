# Keypoint pairwise judge pipeline

Two-step pipeline to build pairwise pose-comparison data with error-type metadata and multiple encodings (A/B as images or text).

## 1. Add error types: `2rinse_error_type.py`

**Input:** `../charles/MLLM-as-a-Judge/pose/selected/selected_pairwise.json` (external).

**Output (this repo):**
- `selected_pairwise_error_type.json` — same entries plus `error_type_a`, `error_type_b`, `error_type_per_person_a`, `error_type_per_person_b` (good / jitter / miss / inversion / swap / over instance / less instance), using [coco-analyze](https://github.com/matteorr/coco-analyze) when built.

**Run:** From this directory, `python 2rinse_error_type.py`. If coco-analyze fails to load, the script will prompt to fall back to the in-script implementation.

---

## 2. Sample, encode, permute: `rinse_pairwise_encode.py`

**Input (this repo):**
- `selected_pairwise_error_type.json` (from step 1).

**Output (this repo):**
- `pairwise_encoded_480.json` — 480 pairwise items (180 by error type + 300 random), each expanded into 4 encoding variants (a, b, c, d). Fields include `answer`, `metadata.encoding`, `metadata.synthetic`, `metadata.synthetic_error_type`, etc.
- `media/encode_same_color/` — encoding **a**: skeletons in one color (green).
- `media/encode_per_person_color/` — encoding **b**: one color per person.
- `media/encode_per_body_part/` — encoding **c**: one color per body part. Encoding **d** is JSON text in the prompt (no extra images).

**Run:** From this directory, `python rinse_pairwise_encode.py`.

---

## Summary

| Step | Script | Main output (paths relative to `keypoint/`) |
|------|--------|---------------------------------------------|
| 1 | `2rinse_error_type.py` | `selected_pairwise_error_type.json` |
| 2 | `rinse_pairwise_encode.py` | `pairwise_encoded_480.json`, `media/encode_same_color/`, `media/encode_per_person_color/`, `media/encode_per_body_part/` |

Run step 1 first, then step 2.
