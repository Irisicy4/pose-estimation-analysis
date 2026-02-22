# pairwise_encoded_480.json 数据质量检查报告

## 概述

对 `pairwise_encoded_480.json` 中的 **457 个 base pair**（1828 条 entry，每个 base pair 含 4 种 encoding 变体）进行了自动化质量检查。

检查覆盖 8 个维度：字段完整性、answer 与 score 方向一致性、cross-encoding 一致性、encoding 格式、media 路径格式、error type 启发式交叉验证、instance type 合理性。

### 结论

- **407 / 457 个 base pair 通过全部自动检查（89%）**
- 50 个 base pair 存在问题，但问题类型集中、可解释，且不影响从通过检查的 407 个中抽样 200-300 个
- **无系统性数据生成 bug**：answer 方向、score 一致性、encoding 格式、cross-encoding 字段同步等核心检查均为 0 错误

---

## 检查结果汇总

| 检查项 | 有问题的 entry 数 | 占比 |
|--------|-----------------|------|
| 字段完整性 | 0 | 0% |
| answer 与 score 方向一致性 | 0 | 0% |
| cross-encoding 字段一致性 | 0 | 0% |
| encoding 格式正确性 | 0 | 0% |
| media 路径格式 | 0 | 0% |
| **error type 启发式交叉验证** | **164** | **9.0%** |
| **instance type 合理性** | **36** | **2.0%** |

---

## 有问题的 50 个 base pair 分类

### 类型 A：error_type=miss 但 OKS score > 0.85（21 个 base pair）

**涉及数据集**：coco300、cocowb300

**现象**：error_type 被标为 "miss"（关键点大幅偏移），但整体 OKS score 很高（0.86–0.95），两者看似矛盾。

**原因**：error_type 取自 per-person 的 worst keypoint 分类——即使一个人只有 1 个 keypoint 被判为 miss，整个人也会被标为 miss。而 OKS 是所有 keypoint 的平均值，其余 keypoint 表现好就能拉高均值。这不是代码 bug，而是 worst-first 聚合策略与平均 OKS 之间的天然张力。

**影响**：这些样本在视觉上差异很小（高 OKS 意味着整体质量好），但 error_type 标签偏严格。对 MLLM 评测而言，这类样本类似于 "noise"——模型可能很难从视觉上区分出 miss。

**预期处理**：可将 error_type 从 `miss` 改为 `jitter`（因 OKS > 0.85 说明整体质量接近 "good"，视觉表现更接近轻微偏移），或保留原标签并在分析时将其视为 borderline case。

### 类型 B：error_type=over instance 但 per_person 只有 1 条（20 个 base pair）

**涉及数据集**：全部来自 mpii300

**现象**：error_type 标为 "over instance"（模型多检了人），但 per_person 列表只有 1 条记录，说明模型实际只检测到 1 人。

**原因**：mpii300 数据集的 GT 标注可能未正确加载（标注数 n_gt=0），导致即使只检测到 1 人也被判定为 n_pred > n_gt → "over instance"。这是数据源（MPII GT 标注格式）的兼容性问题。

**影响**：这 20 个 base pair 的 error_type 标签不可信。

**预期处理**：将 error_type 改为空字符串 `""`（标记为未知），或直接排除。

### 类型 C：instance_type=1C1I 但模型检测到多人（9 个 base pair）

**涉及数据集**：主要为 mpii300 + openpose，1 个 cocowb300

**现象**：instance_type 标为 "1C1I"（单人图），但 error_type_per_person 列表包含多条记录（最多 9 条），说明模型检测到了多个人。

**原因**：instance_type 基于 GT 标注的人数确定。MPII 数据集倾向于只标注主要人物，但图中可能有其他人，openpose 等模型会将背景人物也检测出来。

**影响**：instance_type 标签不准确（实际场景是多人但标为单人），可能影响后续按 instance_type 做的分组分析。

**预期处理**：将 instance_type 从 `1C1I` 改为 `1CnI`。

---

## 可直接使用的数据

从通过检查的 407 个 base pair 中，已通过分层抽样（按 dataset × instance_type 分层）选出 **300 个 base pair**（= 1200 条 entry），分布如下：

| 维度 | OK 池 (407) | 选中 (300) | 采样率 |
|------|-----------|----------|--------|
| synthetic | 148 | 148 | 100%（全部保留） |
| random | 259 | 152 | 59% |
| coco300 | 207 | 153 | 74% |
| cocowb300 | 176 | 133 | 76% |
| mpii300 | 24 | 14 | 58% |
| 1CnI（多人） | 296 | 221 | 75% |
| 1C1I（单人） | 111 | 79 | 71% |

Synthetic error type 分布：jitter=30, inversion=30, swap=30, less instance=30, miss=21, over instance=7。

**说明**：
- miss 的 synthetic 样本从 30 降到 21（9 个属于类型 A 被排除）
- over instance 的 synthetic 仅 7 个（原始数据中符合条件的样本就少）
- mpii300 占比相对较低（原始数据中 mpii300 就较少，且类型 B/C 问题集中在该数据集）

---

## 后续选项

1. **直接使用当前 300 个 base pair**：核心数据质量指标（answer 方向、score 一致性、encoding 格式）均无错误，可直接用于 MLLM 评测。代价是 miss 和 over instance 的 synthetic 样本偏少。

2. **人工矫正后扩充**：对 50 个问题 base pair 做上述修正（类型 B/C 可脚本自动修正，类型 A 需人工判断或统一改为 jitter），修正后重新验证和抽样，可获得更完整的数据集。

3. **混合方案**：先用当前 300 个跑初步实验，同时并行处理矫正工作，后续替换为更完整的版本。
