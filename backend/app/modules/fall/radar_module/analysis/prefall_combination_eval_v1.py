"""纯雷达 pre-fall 组合特征验证（repeat-level，pilot）。

目标
----
在现有 4 类×5 次真机 repeat 数据上，验证哪些纯雷达特征组合最有潜力。
严格限制：
- 统计单位 = repeat（不把连续帧当独立样本）
- 不训练 TCN、不引入新采集、不宣称多人泛化/真实家庭性能/最终部署
- 只允许得出"单受试者受控 pilot 中哪些特征组合最有潜力"

组合
----
A. early-only:    height_range(early) + x_range(early)
B. early+process: A + drift_xy_1p0s(middle/late)

模型对比
----
1. 单特征 height_range(early) —— 基线
2. Logistic Regression（repeat-level，高正则，LORO-CV）
3. LightGBM（小树 + 正则，LORO-CV）

困难对比（主要）
----
- controlled_forward_fall vs fast_sitting
- controlled_forward_fall vs forward_instability_recovery
参考（不主推）：fall vs standing

评估输出
----
- repeat-level AUROC / PR-AUC（LORO OOF 分数）
- leave-one-repeat-out 结果
- 每个 repeat 的 OOF 预测分数（识别拖后腿 repeat）
- 组合 vs 单特征 height_range 的稳定性对比

依赖：scikit-learn、lightgbm（可选）。若缺失，自动降级为
单特征 + 简单阈值/组合评分。

Version: radar_prefall_combination_eval_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FALL = "controlled_forward_fall"
FAST_SITTING = "fast_sitting"
INSTABILITY = "forward_instability_recovery"
STANDING = "standing"

PRIMARY_COMPARISONS = [
    (FALL, FAST_SITTING),
    (FALL, INSTABILITY),
]
REFERENCE_COMPARISONS = [(FALL, STANDING)]

# 组合特征定义（重点，不含 x_range；x_range 仅作 orientation-dependent 辅助）
COMBOS = {
    "HR": ["height_range"],
    "HR_drift": ["height_range", "drift_xy_1p0s"],
}
# x_range 是 orientation-dependent（依赖雷达相对人体朝向），仅辅助实验
ORIENTATION_AUX_COMBOS = {
    "HR_xrange": ["height_range", "x_range"],
}
SINGLE_FEATURE = "height_range"
X_RANGE_ORIENTATION_DEPENDENT = True

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import (  # type: ignore
        average_precision_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

try:
    import lightgbm  # type: ignore

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAS_LIGHTGBM = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_repeat_stages(stage_jsonl: Path) -> dict[str, list[dict[str, Any]]]:
    """加载 per_repeat_stage_features.jsonl，返回 action -> repeats。"""
    rows = _read_jsonl(stage_jsonl)
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_action.setdefault(row["action_name"], []).append(row)
    return by_action


def build_feature_matrix(
    by_action: dict[str, list[dict[str, Any]]],
    pos_action: str,
    neg_action: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构造 pos vs neg 的特征矩阵。

    特征约定：
    - height_range/x_range: 取 early 阶段中位数
    - drift_xy_1p0s: 取 middle/late 过程值（两阶段中位数的均值）
    """
    pos = by_action.get(pos_action, [])
    neg = by_action.get(neg_action, [])
    rows = []
    labels = []
    repeat_ids = []
    for label, repeats in ((1, pos), (0, neg)):
        for rep in repeats:
            feats = {}
            early = rep["stages"].get("early", {})
            middle = rep["stages"].get("middle", {})
            late = rep["stages"].get("late", {})
            feats["height_range"] = early.get("height_range", float("nan"))
            feats["x_range"] = early.get("x_range", float("nan"))
            drift_mid = middle.get("drift_xy_1p0s", float("nan"))
            drift_late = late.get("drift_xy_1p0s", float("nan"))
            feats["drift_xy_1p0s"] = float(np.nanmean([drift_mid, drift_late]))
            rows.append(feats)
            labels.append(label)
            repeat_ids.append(rep["repeat_id"])
    # 过滤 NaN（应极少）
    names = ["height_range", "x_range", "drift_xy_1p0s"]
    X = np.asarray([[row[n] for n in names] for row in rows], dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        X = X[finite_mask]
        y = y[finite_mask]
        repeat_ids = [rid for rid, m in zip(repeat_ids, finite_mask) if m]
    return X, y, repeat_ids


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if not HAS_SKLEARN:
        # 简单 AUROC 实现（Mann-Whitney）
        pos = scores[y == 1]
        neg = scores[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        n_pos, n_neg = len(pos), len(neg)
        combined = np.concatenate([pos, neg])
        order = np.argsort(combined)
        ranks = np.empty(combined.size, dtype=np.float64)
        ranks[order] = np.arange(1, combined.size + 1)
        u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))
    return float(roc_auc_score(y, scores))


def _safe_pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if not HAS_SKLEARN:
        # 简单 PR-AUC：从 recall=0, precision=1 起积分（标准做法）
        if len(y) == 0 or int(y.sum()) == 0:
            return float("nan")
        order = np.argsort(-scores)
        y_sorted = y[order]
        precision = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
        recall = np.cumsum(y_sorted) / int(y.sum())
        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[1.0], precision])
        auc = float(np.trapz(precision, recall))
        return auc if np.isfinite(auc) else float("nan")
    return float(average_precision_score(y, scores))


def single_feature_scores(X: np.ndarray, feat_index: int) -> np.ndarray:
    """单特征直接作为分数（用 height_range）。"""
    return X[:, feat_index]


def _lore_cv_lr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-repeat-out Logistic Regression，返回每个 repeat 的 OOF 概率。

    小样本下 C 过高会把预测压到 0.5（分数退化）。这里用适度 C=0.5，
    并在返回后由调用方做退化检测。
    """
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i : i + 1]
        if np.unique(y_train).size < 2:
            oof[i] = 0.5
            continue
        clf = LogisticRegression(
            C=0.5,  # 适度正则；C=0.01 会导致所有预测≈0.5（分数退化）
            max_iter=2000,
            solver="liblinear",
        )
        try:
            clf.fit(X_train, y_train)
            oof[i] = float(clf.predict_proba(X_test)[0, 1])
        except Exception:
            oof[i] = 0.5
    return oof


def zscore_combo_scores(X: np.ndarray, feat_indices: list[int]) -> np.ndarray:
    """无参数组合基线：所选特征 z-score 平均（不拟合，公平对比单特征）。

    **注意**：本函数使用全数据 mean/std 归一化，仅用于非 LORO 的
    诊断/可视化。**严格评估必须用 zscore_combo_scores_loro**，
    每折只用训练 repeat 计算统计量，避免 test 信息泄漏。
    """
    scores = np.zeros(X.shape[0], dtype=np.float64)
    for idx in feat_indices:
        col = X[:, idx]
        std = np.nanstd(col)
        if std > 0 and np.isfinite(std):
            scores = scores + (col - np.nanmean(col)) / std
    return scores


def zscore_combo_scores_loro(
    X: np.ndarray,
    y: np.ndarray,
    feat_indices: list[int],
) -> np.ndarray:
    """严格无泄漏的 z-score 组合（LORO）。

    对每个 repeat i：
    - 只用除 i 外的训练 repeat 计算每个特征的 mean/std
    - 方向由训练集 pos 与 neg 的均值差符号确定（保证 fall 侧为高分）
    - 用训练统计量归一化 test repeat 并求和

    返回每个 repeat 的 OOF 组合分数（高=fall 侧）。
    """
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        X_train = X[train_mask]
        y_train = y[train_mask]

        score_i = 0.0
        for idx in feat_indices:
            col_train = X_train[:, idx]
            if np.isfinite(col_train).sum() < 3:
                continue
            mean = np.nanmean(col_train)
            std = np.nanstd(col_train)
            if not (std > 0 and np.isfinite(std)):
                continue
            # 方向：训练集 pos 与 neg 均值差，fall 高分
            pos_mean = np.nanmean(col_train[y_train == 1])
            neg_mean = np.nanmean(col_train[y_train == 0])
            direction = 1.0 if pos_mean >= neg_mean else -1.0
            val = X[i, idx]
            if np.isfinite(val):
                score_i += direction * (val - mean) / std
        oof[i] = score_i
    return oof


def _score_degraded(scores: np.ndarray) -> bool:
    """检测 OOF 分数是否退化（几乎恒定，AUROC 无意义）。"""
    finite = scores[np.isfinite(scores)]
    if finite.size < 2:
        return True
    return bool(float(np.nanstd(finite)) < 1e-6)


def _lore_cv_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Leave-one-repeat-out LightGBM，返回每个 repeat 的 OOF 概率。"""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i : i + 1]
        if np.unique(y_train).size < 2:
            oof[i] = 0.5
            continue
        model = lightgbm.LGBMClassifier(
            n_estimators=20,
            max_depth=2,
            num_leaves=4,
            learning_rate=0.1,
            min_child_samples=2,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )
        try:
            model.fit(X_train, y_train)
            oof[i] = float(model.predict_proba(X_test)[0, 1])
        except Exception:
            oof[i] = 0.5
    return oof


def evaluate_pair(
    by_action: dict[str, list[dict[str, Any]]],
    pos_action: str,
    neg_action: str,
) -> dict[str, Any]:
    X, y, repeat_ids = build_feature_matrix(by_action, pos_action, neg_action)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())

    result: dict[str, Any] = {
        "pos_action": pos_action,
        "neg_action": neg_action,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "repeat_ids": repeat_ids,
        "models": {},
    }
    if n_pos < 3 or n_neg < 3:
        result["error"] = "insufficient repeats"
        return result

    # 单特征 height_range（原始值作分数，无统计量拟合，天然无泄漏）
    hr_idx = 0
    hr_scores = single_feature_scores(X, hr_idx)
    result["models"]["single_height_range"] = {
        "auroc": _safe_auroc(y, hr_scores),
        "pr_auc": _safe_pr_auc(y, hr_scores),
        "oof_scores": [float(v) for v in hr_scores],
        "leakage_free": True,
    }

    # 重点组合（无泄漏 LORO）
    feature_order = ["height_range", "x_range", "drift_xy_1p0s"]
    for combo_name, feat_names in COMBOS.items():
        idxs = [feature_order.index(f) for f in feat_names]
        combo: dict[str, Any] = {"features": feat_names}

        # 严格无泄漏 z-score 组合（每折只用训练 repeat 的 mean/std/方向）
        zc_loro = zscore_combo_scores_loro(X, y, idxs)
        combo["zscore_loro"] = {
            "auroc": _safe_auroc(y, zc_loro),
            "pr_auc": _safe_pr_auc(y, zc_loro),
            "oof_scores": [float(v) for v in zc_loro],
            "degraded": _score_degraded(zc_loro),
            "leakage_free": True,
        }

        # LR（LORO 内部每折只拟合训练集，本身无泄漏）
        if HAS_SKLEARN:
            Xc = X[:, idxs]
            lr_oof = _lore_cv_lr(Xc, y)
            combo["logistic_regression"] = {
                "auroc": _safe_auroc(y, lr_oof),
                "pr_auc": _safe_pr_auc(y, lr_oof),
                "oof_scores": [float(v) for v in lr_oof],
                "degraded": _score_degraded(lr_oof),
                "leakage_free": True,
            }
        else:
            combo["logistic_regression"] = {"error": "sklearn not available"}

        # LightGBM
        if HAS_LIGHTGBM:
            Xc = X[:, idxs]
            lgb_oof = _lore_cv_lgbm(Xc, y)
            combo["lightgbm"] = {
                "auroc": _safe_auroc(y, lgb_oof),
                "pr_auc": _safe_pr_auc(y, lgb_oof),
                "oof_scores": [float(v) for v in lgb_oof],
                "degraded": _score_degraded(lgb_oof),
                "leakage_free": True,
            }
        else:
            combo["lightgbm"] = {"error": "lightgbm not available"}

        result["models"][combo_name] = combo

    # x_range 单特征（orientation-dependent，仅辅助）
    xr_scores = single_feature_scores(X, 1)
    result["models"]["single_x_range"] = {
        "auroc": _safe_auroc(y, xr_scores),
        "pr_auc": _safe_pr_auc(y, xr_scores),
        "oof_scores": [float(v) for v in xr_scores],
        "leakage_free": True,
        "orientation_dependent": True,
    }

    # orientation-dependent 辅助组合（含 x_range）
    for combo_name, feat_names in ORIENTATION_AUX_COMBOS.items():
        idxs = [feature_order.index(f) for f in feat_names]
        combo = {"features": feat_names, "orientation_dependent": True}
        zc_loro = zscore_combo_scores_loro(X, y, idxs)
        combo["zscore_loro"] = {
            "auroc": _safe_auroc(y, zc_loro),
            "pr_auc": _safe_pr_auc(y, zc_loro),
            "oof_scores": [float(v) for v in zc_loro],
            "degraded": _score_degraded(zc_loro),
            "leakage_free": True,
        }
        result["models"][combo_name] = combo

    return result


def find_weak_repeats(
    result: dict[str, Any],
    *,
    model_key: str,
    sub_model: str = "zscore_loro",
) -> list[dict[str, Any]]:
    """找拖后腿的 repeat：模型 OOF 分数与真标签差异最大。

    model_key 是组合（HR/HR_drift/HR_xrange），子模型可选 zscore_loro /
    logistic_regression / lightgbm。默认用 zscore_loro（无泄漏，无参数）。
    """
    combo = result["models"].get(model_key)
    if not combo:
        return []
    model = combo.get(sub_model, {})
    if not model or "oof_scores" not in model:
        return []
    repeat_ids = result["repeat_ids"]
    y = np.asarray(
        [1 if rid.startswith(FALL) else 0 for rid in repeat_ids],
        dtype=np.int64,
    )
    scores = np.asarray(model["oof_scores"], dtype=np.float64)
    weak = []
    for rid, true, score in zip(repeat_ids, y, scores):
        # fall(1) 应有高分，非fall(0) 应有低分；偏差 = 真标签与分数的矛盾
        deviation = abs(score - true)
        weak.append({
            "repeat_id": rid,
            "true_label": int(true),
            "oof_score": float(score),
            "deviation": float(deviation),
        })
    weak.sort(key=lambda d: -d["deviation"])
    return weak[:4]


def _model_line(label: str, model: dict[str, Any], indent: bool = False) -> str:
    """把模型 dict 渲染为表格行。"""
    if "error" in model:
        return f"| {label} | n/a | n/a | 不可用 |"
    auroc = model.get("auroc", float("nan"))
    pr = model.get("pr_auc", float("nan"))
    degraded = model.get("degraded", False)
    flag = " ⚠️退化" if degraded else ""
    return f"| {label} | {auroc:.3f} | {pr:.3f} |{flag} |"


def build_report(all_results: dict[str, Any]) -> str:
    lines = [
        "# 纯雷达 pre-fall 组合特征验证（repeat-level，无泄漏 LORO）",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 方法与限制",
        "",
        "- 统计单位 = repeat（4 类 × 5 repeat，每 repeat 一个样本）",
        "- 不训练 TCN；不做多人泛化/真实家庭/最终部署宣称",
        "- 单受试者受控 pilot，n=5+5，结果表述为'未观察到排序错误'而非'完美区分'",
        "- **无泄漏 LORO**：z-score 组合每折只用训练 repeat 计算 mean/std/方向",
        "- 重点组合：HR（height_range）、HR_drift（+drift_xy_1p0s）",
        "- x_range 标记 orientation-dependent（依赖雷达相对人体朝向），仅辅助",
        "- ⚠️退化 = OOF 分数几乎恒定，AUROC 无意义（小样本 ML 常见）",
        "",
        "## 依赖状态",
        f"- scikit-learn: {'OK' if HAS_SKLEARN else '缺失（降级为简单AUROC/PR）'}",
        f"- lightgbm: {'OK' if HAS_LIGHTGBM else '缺失（跳过LightGBM）'}",
        "",
    ]
    for key, comparison in all_results.items():
        pos, neg = comparison.get("pos_action"), comparison.get("neg_action")
        if "error" in comparison:
            lines.append(f"## {pos} vs {neg}: {comparison['error']}")
            continue
        lines.append(f"## {pos} vs {neg} (n={comparison['n_pos']}/{comparison['n_neg']})")
        lines.append("")
        lines.append("| model | AUROC | PR-AUC | 备注 |")
        lines.append("|-------|-------|--------|------|")

        hr = comparison["models"].get("single_height_range", {})
        if "auroc" in hr:
            lines.append(_model_line("**单特征 height_range**", hr))

        for mkey in ["HR", "HR_drift"]:
            model = comparison["models"][mkey]
            feat = model.get("features", [])
            label = f"**{mkey}** ({'+'.join(feat)})"
            zc = model.get("zscore_loro", {})
            if "auroc" in zc:
                lines.append(_model_line(label + " [zscore-LORO]", zc))
            lr = model.get("logistic_regression", {})
            if "auroc" in lr or "error" in lr:
                lines.append(_model_line(label + " [LR]", lr))
            lgb = model.get("lightgbm", {})
            if "auroc" in lgb or "error" in lgb:
                lines.append(_model_line(label + " [LightGBM]", lgb))

        # orientation-dependent 辅助
        xr = comparison["models"].get("single_x_range", {})
        if "auroc" in xr:
            lines.append(_model_line("单特征 x_range ⚠️orientation", xr))
        for mkey in ["HR_xrange"]:
            model = comparison["models"].get(mkey)
            if not model:
                continue
            feat = model.get("features", [])
            zc = model.get("zscore_loro", {})
            if "auroc" in zc:
                lines.append(_model_line(
                    f"{mkey} ({'+'.join(feat)}) ⚠️orientation [zscore-LORO]", zc))

        # 弱 repeat（重点组合）
        for mkey in ["HR", "HR_drift"]:
            weak = find_weak_repeats(comparison, model_key=mkey)
            if weak:
                lines.append("")
                lines.append(f"拖后腿 repeat（{mkey}）：")
                for w in weak:
                    lines.append(
                        f"- {w['repeat_id']}: 真={w['true_label']} "
                        f"OOF={w['oof_score']:.3f} dev={w['deviation']:.3f}"
                    )
        lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeat-level combination feature eval for pre-fall pilot."
    )
    parser.add_argument("--stage-jsonl", type=Path, required=True,
                        help="per_repeat_stage_features.jsonl from stage eval")
    parser.add_argument("--output-root", type=Path,
                        default=Path("reports/prefall_combination_eval_v1"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    by_action = load_repeat_stages(args.stage_jsonl)

    results: dict[str, Any] = {}
    for pos, neg in PRIMARY_COMPARISONS + REFERENCE_COMPARISONS:
        results[f"{pos}_vs_{neg}"] = evaluate_pair(by_action, pos, neg)

    out_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        build_report(results), encoding="utf-8"
    )
    print(f"reports written to {out_dir}")
    print(json.dumps({
        k: {
            "auroc": v["models"].get("single_height_range", {}).get("auroc"),
        } for k, v in results.items()
    }, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
