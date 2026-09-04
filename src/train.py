"""
Model selection, imbalance handling, and hyperparameter tuning for the
return-risk scorer.

Pipeline of decisions (each stage's output feeds the next; nothing here ever
touches the held-out test set - that happens once, in evaluate.py):

  1. Compare 3 model families (Logistic Regression, Random Forest, LightGBM)
     via 5-fold stratified CV, all using class_weight='balanced' so the
     comparison isolates "which model family" from "which imbalance strategy".
  2. Take the CV PR-AUC winner and, for THAT family only, compare
     class_weight vs SMOTE vs SMOTE-Tomek (resampling done inside each CV
     fold via imblearn's Pipeline, so no synthetic near-duplicate ever
     leaks across a fold boundary).
  3. Tune the winning (family, imbalance strategy) combination with Optuna
     (30 trials, TPE sampler, maximizing mean CV PR-AUC).
  4. Refit the tuned pipeline on the FULL training set and persist it.

Outputs:
  reports/model_comparison.csv       - stage 1 results
  reports/imbalance_comparison.csv   - stage 2 results
  reports/optuna_trials.csv          - stage 3 trial history
  reports/model_selection_summary.json - final chosen config + CV scores
  models/final_model.joblib          - fitted pipeline (trained on X_train only)
"""
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import TARGET_ENCODE_COLS, TargetMeanEncoder, load_and_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_SPLITS = 5
N_OPTUNA_TRIALS = 30
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

DEFAULT_PARAMS = {
    "logreg": {},
    "rf": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
    "lgbm": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
}


def build_pipeline(family: str, imbalance: str = "class_weight", params: dict | None = None):
    params = {**DEFAULT_PARAMS.get(family, {}), **(params or {})}
    steps = [("target_enc", TargetMeanEncoder(TARGET_ENCODE_COLS))]

    needs_impute = family in ("logreg", "rf") or imbalance in ("smote", "smote_tomek")
    if needs_impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if family == "logreg":
        steps.append(("scale", StandardScaler()))

    if imbalance == "smote":
        steps.append(("resample", SMOTE(random_state=SEED)))
    elif imbalance == "smote_tomek":
        steps.append(("resample", SMOTETomek(random_state=SEED)))

    class_weight = "balanced" if imbalance == "class_weight" else None
    if family == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=SEED, **params)
    elif family == "rf":
        clf = RandomForestClassifier(class_weight=class_weight, random_state=SEED, n_jobs=-1, **params)
    elif family == "lgbm":
        clf = LGBMClassifier(class_weight=class_weight, random_state=SEED, n_jobs=-1, verbosity=-1, **params)
    else:
        raise ValueError(f"Unknown family: {family}")
    steps.append(("clf", clf))

    PipelineCls = ImbPipeline if imbalance in ("smote", "smote_tomek") else SkPipeline
    return PipelineCls(steps)


def cv_scores(pipeline, X, y, n_splits=N_SPLITS) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    result = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    return {
        "roc_auc_mean": result["test_roc_auc"].mean(),
        "roc_auc_std": result["test_roc_auc"].std(),
        "pr_auc_mean": result["test_pr_auc"].mean(),
        "pr_auc_std": result["test_pr_auc"].std(),
    }


def stage1_model_comparison(X_train, y_train) -> pd.DataFrame:
    print("=== Stage 1: model family comparison (5-fold CV, class_weight='balanced') ===")
    rows = []
    for family in ["logreg", "rf", "lgbm"]:
        t0 = time.time()
        pipe = build_pipeline(family, imbalance="class_weight")
        scores = cv_scores(pipe, X_train, y_train)
        scores["family"] = family
        scores["seconds"] = round(time.time() - t0, 1)
        rows.append(scores)
        print(f"  {family:8s}  ROC-AUC {scores['roc_auc_mean']:.4f} (+/-{scores['roc_auc_std']:.4f})  "
              f"PR-AUC {scores['pr_auc_mean']:.4f} (+/-{scores['pr_auc_std']:.4f})  [{scores['seconds']}s]")
    df = pd.DataFrame(rows)[["family", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "seconds"]]
    df.to_csv("reports/model_comparison.csv", index=False)
    return df


def stage2_imbalance_comparison(winner_family: str, X_train, y_train) -> pd.DataFrame:
    print(f"\n=== Stage 2: imbalance handling comparison for winner='{winner_family}' ===")
    rows = []
    for imbalance in ["class_weight", "smote", "smote_tomek"]:
        t0 = time.time()
        pipe = build_pipeline(winner_family, imbalance=imbalance)
        scores = cv_scores(pipe, X_train, y_train)
        scores["imbalance"] = imbalance
        scores["seconds"] = round(time.time() - t0, 1)
        rows.append(scores)
        print(f"  {imbalance:13s}  ROC-AUC {scores['roc_auc_mean']:.4f} (+/-{scores['roc_auc_std']:.4f})  "
              f"PR-AUC {scores['pr_auc_mean']:.4f} (+/-{scores['pr_auc_std']:.4f})  [{scores['seconds']}s]")
    df = pd.DataFrame(rows)[["imbalance", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "seconds"]]
    df.to_csv("reports/imbalance_comparison.csv", index=False)
    return df


def suggest_params(trial: optuna.Trial, family: str) -> dict:
    if family == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if family == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if family == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    raise ValueError(family)


def stage3_hyperparameter_tuning(family: str, imbalance: str, X_train, y_train):
    print(f"\n=== Stage 3: Optuna tuning ({N_OPTUNA_TRIALS} trials) for family='{family}', "
          f"imbalance='{imbalance}' ===")

    def objective(trial):
        params = suggest_params(trial, family)
        pipe = build_pipeline(family, imbalance=imbalance, params=params)
        scores = cv_scores(pipe, X_train, y_train)
        return scores["pr_auc_mean"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv("reports/optuna_trials.csv", index=False)
    print(f"  Best CV PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def main():
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape} (test untouched until evaluate.py)\n")

    comparison_df = stage1_model_comparison(X_train, y_train)
    winner_family = comparison_df.loc[comparison_df["pr_auc_mean"].idxmax(), "family"]
    print(f"\nStage 1 winner (by CV PR-AUC): {winner_family}")

    imbalance_df = stage2_imbalance_comparison(winner_family, X_train, y_train)
    winner_imbalance = imbalance_df.loc[imbalance_df["pr_auc_mean"].idxmax(), "imbalance"]
    baseline_pr_auc = imbalance_df.loc[imbalance_df["imbalance"] == "class_weight", "pr_auc_mean"].iloc[0]
    best_pr_auc = imbalance_df["pr_auc_mean"].max()
    # Keep the simpler class_weight strategy unless resampling wins by a
    # non-trivial margin (avoids swapping in SMOTE for noise-level gains).
    if best_pr_auc - baseline_pr_auc < 0.005:
        winner_imbalance = "class_weight"
    print(f"Stage 2 winner (by CV PR-AUC, >=0.005 margin required to beat class_weight): {winner_imbalance}")

    best_params, best_cv_pr_auc = stage3_hyperparameter_tuning(winner_family, winner_imbalance, X_train, y_train)

    print("\n=== Refitting final pipeline on full training set ===")
    final_pipeline = build_pipeline(winner_family, imbalance=winner_imbalance, params=best_params)
    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, "models/final_model.joblib")

    tuned_scores = cv_scores(build_pipeline(winner_family, imbalance=winner_imbalance, params=best_params),
                              X_train, y_train)

    summary = {
        "model_family": winner_family,
        "imbalance_strategy": winner_imbalance,
        "best_params": best_params,
        "cv_roc_auc_mean": tuned_scores["roc_auc_mean"],
        "cv_roc_auc_std": tuned_scores["roc_auc_std"],
        "cv_pr_auc_mean": tuned_scores["pr_auc_mean"],
        "cv_pr_auc_std": tuned_scores["pr_auc_std"],
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_positive_rate": float(y_train.mean()),
    }
    with open("reports/model_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinal model: {winner_family} + {winner_imbalance}")
    print(f"CV ROC-AUC: {tuned_scores['roc_auc_mean']:.4f}  CV PR-AUC: {tuned_scores['pr_auc_mean']:.4f}")
    print("Saved: models/final_model.joblib, reports/model_selection_summary.json")


if __name__ == "__main__":
    main()
