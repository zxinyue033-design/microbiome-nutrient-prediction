import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score

PROJECT_DIR = "/Users/xinyue/Desktop/shen/glucose"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset_glucose_KO_ml_posneg.csv")

OUT_RESULTS = os.path.join(DATA_DIR, "baseline_results.csv")
OUT_IMPORTANCE = os.path.join(DATA_DIR, "baseline_feature_importance.csv")


def pick_k(n_splits_target: int, y: np.ndarray) -> int:
    vals, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min())
    return max(2, min(n_splits_target, min_class))


def print_scores(name: str, scores: dict):
    def mean_std(arr):
        return float(np.mean(arr)), float(np.std(arr))

    auc_m, auc_s = mean_std(scores["test_roc_auc"])
    ap_m, ap_s = mean_std(scores["test_average_precision"])
    bac_m, bac_s = mean_std(scores["test_balanced_accuracy"])

    print(f"\n=== {name} (CV) ===")
    print(f"ROC-AUC:          {auc_m:.3f} ± {auc_s:.3f}")
    print(f"Avg Precision(AP):{ap_m:.3f} ± {ap_s:.3f}")
    print(f"Balanced Acc:     {bac_m:.3f} ± {bac_s:.3f}")


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    if "species_label" not in df.columns:
        raise ValueError("Expected column 'species_label' in dataset.")

    # 自动识别 KO 特征列（以 K 开头）
    feature_cols = [c for c in df.columns if c.startswith("K")]
    if not feature_cols:
        raise ValueError("No KO feature columns found (columns starting with 'K').")

    X = df[feature_cols].astype(float).values
    y = df["species_label"].astype(int).values

    print("[INFO] Loaded:", DATASET_PATH)
    print("[INFO] Shape:", df.shape)
    print("[INFO] #features:", len(feature_cols))
    print("[INFO] Label distribution:")
    print(df["species_label"].value_counts())

    # CV splits（自动适配小样本）
    k = pick_k(5, y)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    print(f"[INFO] Using StratifiedKFold with n_splits={k}")

    # 指标
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    # Model 1: Logistic Regression
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )),
        ]
    )

    scores_lr = cross_validate(logreg, X, y, cv=cv, scoring=scoring, return_train_score=False)
    print_scores("LogisticRegression", scores_lr)


    # Model 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    scores_rf = cross_validate(rf, X, y, cv=cv, scoring=scoring, return_train_score=False)
    print_scores("RandomForest", scores_rf)

    # 保存 CV 结果
    results_rows = []
    for model_name, scores in [("LogisticRegression", scores_lr), ("RandomForest", scores_rf)]:
        results_rows.append({
            "model": model_name,
            "n_splits": k,
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "ap_mean": float(np.mean(scores["test_average_precision"])),
            "ap_std": float(np.std(scores["test_average_precision"])),
            "balanced_acc_mean": float(np.mean(scores["test_balanced_accuracy"])),
            "balanced_acc_std": float(np.std(scores["test_balanced_accuracy"])),
            "n_samples": int(len(y)),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
            "n_features": int(len(feature_cols)),
        })

    pd.DataFrame(results_rows).to_csv(OUT_RESULTS, index=False)
    print(f"\n[OK] Saved CV summary to: {OUT_RESULTS}")

    # Fit on full dataset for interpretability (not a valid test score)
    print("\n[INFO] Fitting models on FULL data to get feature importance (not evaluation).")

    logreg.fit(X, y)
    lr_coef = logreg.named_steps["clf"].coef_.ravel()

    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "logreg_coef": lr_coef,
        "logreg_abscoef": np.abs(lr_coef),
        "rf_importance": rf_imp
    }).sort_values(by=["rf_importance", "logreg_abscoef"], ascending=False)

    imp_df.to_csv(OUT_IMPORTANCE, index=False)
    print(f"[OK] Saved feature importance to: {OUT_IMPORTANCE}")

    print("\nTop 15 features by RF importance / |LR coef|:")
    print(imp_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()