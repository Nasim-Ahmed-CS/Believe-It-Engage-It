from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class Config:
    base_dir: Path
    min_rows_per_window: int = 10
    risk_mode: str = "low"
    risk_cutoff: float = 3.0
    escalation_threshold: Optional[float] = None
    target_recall_for_escalation: float = 0.85
    min_escalated_train: int = 20
    min_escalation_rate: float = 0.20
    test_size: float = 0.30
    random_state: int = 42


def normalize_participant_id(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    match = re.search(r"(\d+)", str(raw))
    return match.group(1) if match else None


def load_label_map(self_report_csv: Path) -> Dict[Tuple[str, int], float]:
    df = pd.read_csv(self_report_csv, skiprows=1)
    score_type = df["Score Type"].astype(str).str.strip().str.lower()
    df = df[score_type == "engagement"].copy()
    if df.empty:
        raise ValueError(f"No engagement rows found in {self_report_csv}")

    df["pid"] = df["Participant"].apply(normalize_participant_id)
    minute_cols = [c for c in df.columns if re.match(r"^Min\s+\d+$", str(c).strip())]
    minute_cols = sorted(minute_cols, key=lambda c: int(re.search(r"\d+", c).group()))

    label_map: Dict[Tuple[str, int], float] = {}
    for _, row in df.iterrows():
        pid = row["pid"]
        if pid is None:
            continue
        for col in minute_cols:
            val = row[col]
            if pd.isna(val):
                continue
            minute = int(re.search(r"\d+", col).group())
            label_map[(pid, minute)] = float(val)
    return label_map


def aggregate_window(window: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in feature_cols:
        arr = window[col].to_numpy(dtype=float)
        out[f"{col}__mean"] = float(np.nanmean(arr))
        out[f"{col}__std"] = float(np.nanstd(arr))
        out[f"{col}__min"] = float(np.nanmin(arr))
        out[f"{col}__max"] = float(np.nanmax(arr))
    return out


def build_feature_table(
    processed_dir: Path,
    file_glob: str,
    label_map: Dict[Tuple[str, int], float],
    min_rows_per_window: int,
    condition: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    files = sorted(processed_dir.glob(file_glob))

    for fp in files:
        pid_match = re.search(r"(\d+)\.csv$", fp.name)
        if not pid_match:
            continue
        pid = pid_match.group(1)

        df = pd.read_csv(fp)
        if "TimestampRaw" not in df.columns:
            continue

        feature_cols = [c for c in df.columns if c != "TimestampRaw"]
        if not feature_cols:
            continue

        df = df.sort_values("TimestampRaw").reset_index(drop=True)
        t0 = float(df["TimestampRaw"].iloc[0])
        df["t_rel"] = df["TimestampRaw"] - t0
        max_t = float(df["t_rel"].max())

        participant_minutes = sorted([m for (p, m) in label_map.keys() if p == pid])
        for minute in participant_minutes:
            end_t = minute * 60
            start_t = end_t - 120
            if end_t > max_t + 1:
                continue

            window = df[(df["t_rel"] >= start_t) & (df["t_rel"] < end_t)]
            if len(window) < min_rows_per_window:
                continue

            feats = aggregate_window(window, feature_cols)
            score = float(label_map[(pid, minute)])
            record: Dict[str, object] = {
                "participant": pid,
                "minute": minute,
                "engagement_score": score,
                "condition": condition,
                "source_file": fp.name,
            }
            record.update(feats)
            rows.append(record)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["condition_multimodal"] = (out["condition"] == "multimodal").astype(int)
    return out


def select_stage1_stage2_columns(feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    low_cost_bases = [
        "brightness_flicker",
        "optical_flow",
        "scene_cuts",
        "spectral_entropy",
        "luminance",
        "contrast",
    ]

    stage1_cols: List[str] = []
    for b in low_cost_bases:
        c = f"{b}__mean"
        if c in feature_columns:
            stage1_cols.append(c)

    if "condition_multimodal" in feature_columns:
        stage1_cols.append("condition_multimodal")

    if not stage1_cols:
        fallback = [c for c in feature_columns if c.endswith("__mean")]
        stage1_cols = sorted(fallback)[:8] if fallback else sorted(feature_columns)[:8]

    stage2_cols = sorted(feature_columns)
    return stage1_cols, stage2_cols


def choose_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> float:
    candidates = np.unique(np.round(y_prob, 6))
    candidates = np.sort(candidates)

    best_threshold = 0.5
    best_precision = -1.0
    for th in candidates:
        y_pred = (y_prob >= th).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall:
            prec = precision_score(y_true, y_pred, zero_division=0)
            if prec > best_precision:
                best_precision = prec
                best_threshold = float(th)

    if best_precision < 0:
        return 0.5
    return best_threshold


def choose_threshold_for_budget(y_prob: np.ndarray, min_count: int, min_rate: float) -> float:
    n = len(y_prob)
    need = max(min_count, int(np.ceil(n * min_rate)))
    need = min(max(1, need), n)
    sorted_desc = np.sort(y_prob)[::-1]
    idx = max(0, need - 1)
    return float(sorted_desc[idx])


def compute_target(scores: np.ndarray, mode: str, cutoff: float) -> np.ndarray:
    if mode == "low":
        return (scores <= cutoff).astype(int)
    if mode == "high":
        return (scores >= cutoff).astype(int)
    raise ValueError("risk_mode must be 'low' or 'high'")


def safe_metric(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def train_rf(X: np.ndarray, y: np.ndarray, random_state: int, n_estimators: int = 500) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    model.fit(X, y)
    return model


def run(cfg: Config) -> None:
    mm_dir = cfg.base_dir / "MultiModalAttack"
    txt_dir = cfg.base_dir / "TextualAttack"

    mm_labels = load_label_map(mm_dir / "Trust and Engagement Self Report_Multi_Modal.csv")
    txt_labels = load_label_map(txt_dir / "Trust and Engagement Self Report_TextOnly.csv")

    mm_table = build_feature_table(
        processed_dir=mm_dir / "Processed Data",
        file_glob="DI_Engagement_False_Info_*.csv",
        label_map=mm_labels,
        min_rows_per_window=cfg.min_rows_per_window,
        condition="multimodal",
    )
    txt_table = build_feature_table(
        processed_dir=txt_dir / "Processed Data",
        file_glob="DI_Engagement_False_Info_Text_*.csv",
        label_map=txt_labels,
        min_rows_per_window=cfg.min_rows_per_window,
        condition="text",
    )

    full = pd.concat([mm_table, txt_table], ignore_index=True)
    if full.empty:
        raise RuntimeError("No samples built. Check paths and input CSVs.")

    non_feature_cols = {"participant", "minute", "engagement_score", "condition", "source_file"}
    feature_cols = [c for c in full.columns if c not in non_feature_cols]
    stage1_cols, stage2_cols = select_stage1_stage2_columns(feature_cols)

    y = compute_target(full["engagement_score"].to_numpy(dtype=float), cfg.risk_mode, cfg.risk_cutoff)
    groups = full["participant"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_idx, test_idx = next(splitter.split(full, y, groups=groups))

    X_stage1 = full[stage1_cols].to_numpy(dtype=float)
    X1_train, X1_test = X_stage1[train_idx], X_stage1[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Need both classes in train and test split. Try changing cutoff or random_state.")

    stage1 = train_rf(X1_train, y_train, cfg.random_state, n_estimators=300)
    stage1_prob_train = stage1.predict_proba(X1_train)[:, 1]
    if cfg.escalation_threshold is None:
        used_threshold = choose_threshold_for_recall(
            y_true=y_train,
            y_prob=stage1_prob_train,
            target_recall=cfg.target_recall_for_escalation,
        )
    else:
        used_threshold = cfg.escalation_threshold

    stage1_prob_test = stage1.predict_proba(X1_test)[:, 1]
    stage1_pred_test = (stage1_prob_test >= 0.5).astype(int)

    stage1_metrics = safe_metric(y_test, stage1_pred_test, stage1_prob_test)
    escalated_train = stage1_prob_train >= used_threshold
    if np.sum(escalated_train) < cfg.min_escalated_train:
        used_threshold = choose_threshold_for_budget(
            y_prob=stage1_prob_train,
            min_count=cfg.min_escalated_train,
            min_rate=cfg.min_escalation_rate,
        )
        escalated_train = stage1_prob_train >= used_threshold

    escalated_test = stage1_prob_test >= used_threshold
    escalation_rate = float(np.mean(escalated_test))

    print("=== Two-Stage Escalation Experiment ===")
    print(f"Samples: {len(full)} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"Risk mode: {cfg.risk_mode} | cutoff: {cfg.risk_cutoff}")
    print(f"Stage 1 features: {len(stage1_cols)}")
    print(f"Stage 2 features: {len(stage2_cols)}")

    print("\n[Stage 1: Lightweight Monitor]")
    for k, v in stage1_metrics.items():
        print(f"{k}: {v}")
    print(f"Escalation threshold used: {used_threshold:.4f}")
    print(f"Escalated train samples: {int(np.sum(escalated_train))}")
    print(f"Escalation rate (test): {escalation_rate:.3f}")

    X_stage2 = full[stage2_cols].to_numpy(dtype=float)
    X2_train, X2_test = X_stage2[train_idx], X_stage2[test_idx]

    y2_train = y_train[escalated_train]
    if len(np.unique(y2_train)) < 2:
        print("\n[Stage 2: High-Resolution Tracker]")
        print("Escalated training set has one class only; Stage 2 skipped.")
        return

    X2_train_stack = np.column_stack([X2_train[escalated_train], stage1_prob_train[escalated_train]])
    stage2 = train_rf(X2_train_stack, y2_train, cfg.random_state, n_estimators=800)

    y_final_pred = stage1_pred_test.copy()
    y_final_prob = stage1_prob_test.copy()
    if np.any(escalated_test):
        X2_test_stack = np.column_stack([X2_test[escalated_test], stage1_prob_test[escalated_test]])
        s2_prob = stage2.predict_proba(X2_test_stack)[:, 1]
        s2_pred = (s2_prob >= 0.5).astype(int)
        y_final_pred[escalated_test] = s2_pred
        y_final_prob[escalated_test] = s2_prob

        stage2_metrics = safe_metric(y_test[escalated_test], s2_pred, s2_prob)
        print("\n[Stage 2: High-Resolution Tracker]")
        print(f"Escalated test samples: {int(np.sum(escalated_test))}")
        for k, v in stage2_metrics.items():
            print(f"{k}: {v}")
    else:
        print("\n[Stage 2: High-Resolution Tracker]")
        print("No escalated test samples at this threshold; Stage 2 not applied on test set.")

    final_metrics = safe_metric(y_test, y_final_pred, y_final_prob)
    print("\n[Cascade Final Metrics]")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resource-aware two-stage escalation experiment with available features")
    p.add_argument("--base-dir", type=str, default=".", help="Project root directory")
    p.add_argument("--risk-mode", type=str, choices=["low", "high"], default="low")
    p.add_argument("--risk-cutoff", type=float, default=3.0)
    p.add_argument("--escalation-threshold", type=float, default=None)
    p.add_argument("--target-recall-for-escalation", type=float, default=0.85)
    p.add_argument("--min-escalated-train", type=int, default=20)
    p.add_argument("--min-escalation-rate", type=float, default=0.20)
    p.add_argument("--min-rows-per-window", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        base_dir=Path(args.base_dir).resolve(),
        min_rows_per_window=args.min_rows_per_window,
        risk_mode=args.risk_mode,
        risk_cutoff=args.risk_cutoff,
        escalation_threshold=args.escalation_threshold,
        target_recall_for_escalation=args.target_recall_for_escalation,
        min_escalated_train=args.min_escalated_train,
        min_escalation_rate=args.min_escalation_rate,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    run(cfg)
