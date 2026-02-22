# Pairwise Data 质量检查与修复报告

## TL;DR

对 `pairwise_encoded_480.json`（1828 条，457 组 base pair × 4 种 encoding）做了系统性质量检查，发现并修复了以下问题：

1. **yolo8n 预测加载 bug**：keypoints 格式不兼容（nested vs flat），导致 132 组 yolo8n 相关 pair 的 error_type 全部为空 → 已用本地 GT + 预测文件重新计算
2. **error_type 标签错误**：miss+高 OKS（21 组）、mpii300 over_instance GT 异常（20 组）、instance_type 不准（9 组）→ 已修正
3. **新增 instance-aware metric**：`adjusted_score = OKS × instance_recall`，解决 less instance 场景下 OKS 不惩罚漏检的问题 → less instance 的 MLLM 准确率从 40% 提升至 53%

修复后 **453/457 组通过验证**（99.1%），已分层抽样 300 组可直接用于评测。

---

## 新增文件一览

### 核心产出

| 文件 | 说明 |
|------|------|
| `pairwise_encoded_480.json`（根目录，已更新） | 修复后的完整数据，含新 metric 字段 |
| `check/sampled_300.json` | 从 453 组 OK pair 中分层抽样的 300 组 × 4 encoding = 1200 条 |

### 源码改动

| 文件 | 改动 |
|------|------|
| `2rinse_error_type.py`（根目录，已更新） | (1) `load_predictions` 格式 bug fix; (2) 新增 `compute_instance_recall`, `compute_adjusted_score`; (3) `process_entry` 输出新字段 |

### 检查/修复脚本（`check/` 目录）

| 脚本 | 用途 | 何时使用 |
|------|------|----------|
| `validate_pairwise.py` | 自动化数据验证（结构 + 语义） | 每次修改数据后跑一次 |
| `fix_known_issues.py` | 修复已知标签问题 | 从原始数据开始修复时 |
| `recompute_yolo8n_with_gt.py` | 用 GT 重算 yolo8n error_type | 需要 `gt_annotations/` 和 `yolo8n_results/` |
| `apply_adjusted_score.py` | 计算 adjusted_score 并更新 answer | 修复完 error_type 后 |
| `stratified_sample.py` | 分层抽样 | `--target 300` 抽 300 组 |
| `compare_adjusted_accuracy.py` | 对比 OKS vs adjusted 准确率 | 需要 `work_dirs_judge_keypoint/` |
| `update_result_metadata.py` | 更新 inference result.json 的 metadata | 重跑分析图表前 |

### 辅助数据（未上传，需从 server 获取）

以下目录在本地使用过但未上传到 repo，复现完整流程时需要：

| 目录 | 来源 | 用途 |
|------|------|------|
| `gt_annotations/{coco300,cocowb300,mpii300}/annotations.json` | `/raid/charles/MLLM-as-a-Judge/pose/data/` | GT 标注，重算 error_type 需要 |
| `yolo8n_results/{coco300,cocowb300,mpii300}/` | server 上 yolo8n 预测输出 | yolo8n 预测文件 |
| `yolo8n_infer/` | 推理代码 | yolo8n 推理脚本 + 模型权重 |
| `work_dirs_judge_keypoint/` | MLLM inference 结果 | 各 MLLM 的 result.json + 分析图表 |

### 日志

| 目录 | 说明 |
|------|------|
| `check/logs/` | 各步骤的详细修改日志 |

---

## 发现的问题与修复

### Bug 1: yolo8n 预测加载失败（影响 132 组）

**根因**：`2rinse_error_type.py` 的 `load_predictions()` 检查 `len(kp) < 51` 时跳过了 yolo8n，因为 yolo8n 的 keypoints 是嵌套格式 `[[x,y,v]]*17`（长度 17），而代码期望扁平格式 `[x,y,v]*17`（长度 51）。

**后果**：所有 yolo8n 条目的 `error_type_per_person` 为空，n_pred=0，被错误标为 `less instance`。

**修复**：
- `2rinse_error_type.py` 的 `load_predictions`: 新增嵌套格式检测和展开逻辑
- `check/recompute_yolo8n_with_gt.py`: 用本地 GT 标注重新计算 per-keypoint error classification（包含 MPII 16→COCO 17 关键点映射）

### Bug 2: 其他标签问题（影响 50 组）

| 问题 | 数量 | 修复 |
|------|------|------|
| `instance_type` 1C1I 但 per_person 有多人 | 9 组 | → `1CnI` |
| mpii300 `over instance` 但 per_person ≤ 1 | 20 组 | → `""` (unknown) |
| `miss` 但 OKS > 0.85 | 21 组 | → `jitter` |

### 新 Metric: Instance-Aware Adjusted Score

**问题**：OKS 只衡量检测到的 keypoint 质量，不惩罚漏检。一个模型检测到 3/10 个人但很准确，OKS 可能比检测到 10/10 个人但稍有偏移的模型还高。

**方案**：
```
instance_recall = min(n_pred, n_gt) / n_gt
adjusted_score  = OKS × instance_recall
```

**效果**（用已有 MLLM inference 结果验证）：

| 指标 | OKS-based | Adjusted | 变化 |
|------|-----------|----------|------|
| **Overall** | 50.3% | 51.7% | +1.4% |
| **less instance** | 40.0% | **53.2%** | **+13.2%** |

新增 metadata 字段：`n_pred_a`, `n_pred_b`, `n_gt`, `instance_recall_a`, `instance_recall_b`, `adjusted_score_a`, `adjusted_score_b`, `adjusted_score_difference`, `adjusted_answer`

---

## 复现完整修复流程

```bash
# 1. 从原始数据开始
cp backups/pairwise_encoded_480.pre_fix_20260221_161834.json pairwise_encoded_480.json

# 2. 修复已知标签问题
python check/fix_known_issues.py

# 3. 用 GT 重算 yolo8n（需要 gt_annotations/ 和 yolo8n_results/）
python check/recompute_yolo8n_with_gt.py

# 4. 计算 adjusted_score 新 metric
python check/apply_adjusted_score.py

# 5. 验证
python check/validate_pairwise.py

# 6. 分层抽样 300 组
python check/stratified_sample.py --target 300
```

---

## 后续建议

1. **重新生成评测题目**：用 adjusted_score 确定的 answer 在 server 上重跑 `rinse_pairwise_encode.py`，然后重新跑 MLLM inference，预期 less instance 准确率会进一步提升
2. **4 个 swap 样本**：两边都是 swap 且 score 差异 < 0.05，目前已排除在 300 组之外，可保持现状
3. **`2rinse_error_type.py` 部署到 server**：修复格式 bug 后所有模型都能正确加载，新 metric 字段会自动添加
