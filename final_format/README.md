# Keypoint final format

This directory holds the **rinsed** keypoint (pose estimation) dataset in a unified format for MLLM-as-a-judge evaluation: one `images.json` (original images) and one `annotations.json` (one annotation per image×model or image×GT, with keypoint predictions and image-level error types and scores).

The format mirrors the instance-segmentation reference in `ref/` (see `ref/register_instseg.py` and `ref/images.json` / `ref/annotations.json`), adapted for keypoint tasks.

---

## Output files

| File | Description |
|------|-------------|
| `images.json` | List of original images: one record per unique base image (id, file_path, data_source, height, width, etc.). |
| `annotations.json` | List of annotations: one per (image, model) or per (image, gt). Each has `predictions` (list of skeletons), `error_type`, `final_score`, and related fields. |
| `final_images/keypoint/{dataset}/` | Copies of the original images (e.g. `{sample_id}.jpg`) used by `file_path` in `images.json`. |

---

## Images format (`images.json`)

Each element is an image record:

```json
{
  "id": "coco300_000000001000",
  "file_path": "keypoint/coco300/000000001000.jpg",
  "data_source": "coco300",
  "height": 480,
  "width": 640,
  "scene": "",
  "is_crowd": false,
  "is_longtail": false,
  "groundtruth_class": ["person"],
  "groundtruth_class_id": []
}
```

- **id**: `{dataset}_{sample_id}`.
- **file_path**: Path relative to the media root, under `keypoint/{dataset}/`.
- **data_source**: Dataset name (e.g. `coco300`).
- **height**, **width**: From GT annotations when available.

---

## Annotations format (`annotations.json`)

Each element is one annotation: either **ground truth** for an image or **one model’s prediction** on that image.

### Common fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique annotation id (e.g. `kp_0`, `kp_1`, …). |
| `task` | string | Always `"keypoint"`. |
| `image_id` | string | Same as `images[].id` (`{dataset}_{sample_id}`). |
| `coi` | list[string] | Classes of interest; always `["person"]`. |
| `instance_type` | string | `"1C1I"` (one person) or `"1CnI"` (multiple persons). |
| `model_name` | string | Model id (e.g. `dwpose`, `yolo8m`) or `"gt"`. |
| `error_type` | string | Image-level error type (see below). |
| `final_score` | float | Score in [0, 1]; see scoring below. |
| `other_scores` | object | E.g. `{"oks": ..., "avg_oks": ...}`. |
| `predictions_type` | string | Always `"keypoint"`. |
| `predictions` | list | List of skeletons; see below. |

### Error types (`error_type`)

- **gt**: Ground truth.
- **good**: No dominant error.
- **jitter**: Small localization noise.
- **miss**: Keypoint not detected / wrong place.
- **inversion**: Keypoint matched to another keypoint of the same person.
- **swap**: Keypoint matched to another person’s keypoint.
- **less instance**: Fewer predicted persons than GT.
- **over instance**: More predicted persons than GT.

These are the **image-level** aggregated types from the Ronchi & Perona (ICCV 2017) style analysis used in `../2rinse_error_type.py`.

### Predictions (`predictions`)

- **Type**: List of skeletons. One annotation = **all** skeletons for that image for that model (or GT).
- **One skeleton**: COCO 17-keypoint format: 51 floats `[x,y,v, x,y,v, …]` (17×3). Order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle. `v`: visibility (0=missing, 1=occluded, 2=visible).

Example (one annotation with two persons):

```json
"predictions": [
  [162.0, 174.0, 2.0, 167.0, 170.0, 2.0, ... ],
  [423.0, 150.0, 2.0, 428.0, 143.0, 2.0, ... ]
]
```

### Scoring (`final_score`)

- **GT**: `1.0`.
- **Model**:
  - Base = average OKS (avg_oks) from pairwise comparison.
  - Adjustment: add `0.1 * avg_oks` → effective base = `1.1 * avg_oks`.
  - If `error_type` is **less instance** or **over instance**: subtract `0.1` per instance gap:
    - `final_score = 1.1 * avg_oks - 0.1 * |n_pred - n_gt|`
  - Otherwise: `final_score = 1.1 * avg_oks`
  - Values are clamped to `[0, 1]`.

---

## Generating the files: `rinse_giant.py`

The script `rinse_giant.py` builds `images.json` and `annotations.json` from:

1. **Pairwise + error metadata**: `../selected_pairwise_error_type.json` (from `../2rinse_error_type.py`).
2. **Ground truth**: `{CHARLES_POSE_ROOT}/data/{dataset}/annotations.json` (COCO-style).
3. **Model predictions**: `{CHARLES_POSE_ROOT}/results/{model_id}/{dataset}/{sample_id}_keypoints.json`.
4. **Original images**: `{CHARLES_POSE_ROOT}/selected/images/` (copied into `final_images/keypoint/{dataset}/`).

### Requirements

- `../selected_pairwise_error_type.json` must exist (run `../2rinse_error_type.py` first if needed).
- Charles pose data layout:
  - `CHARLES_POSE_ROOT` = `/raid/charles/MLLM-as-a-Judge/pose`
  - `data/{dataset}/annotations.json`, `results/{model_id}/{dataset}/{sample_id}_keypoints.json`, `selected/images/...`

### Run

```bash
cd /raid/icy/aggregate-mllm-as-a-judge/keypoint/final_format
python rinse_giant.py
```

This will:

- Collect unique (dataset, sample_id) and (image, model) metadata from the pairwise JSON.
- Build `images.json` and copy originals into `final_images/keypoint/{dataset}/`.
- Build `annotations.json`: one GT annotation per image (all GT skeletons in `predictions`), one model annotation per (image, model) (all predicted skeletons in `predictions`), with image-level `error_type` and `final_score` as above.

---

## Reference: `ref/`

The `ref/` directory contains the **instance segmentation** counterpart:

- **ref/register_instseg.py**: Builds instance_seg images + annotations from error_annotations.
- **ref/images.json**, **ref/annotations.json**: Example format (same image/annotation structure, different `task` and `predictions_type`/`predictions` content).

For keypoints we use the same high-level schema (images list + annotations list, same field names where applicable), with `task: "keypoint"`, `predictions_type: "keypoint"`, and `predictions` as a list of COCO keypoint vectors per image.
