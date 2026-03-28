# -*- coding: utf-8 -*-
"""
predictive_model.py — Team Coding Rookie, Data Hackathon 2026

Predictive modeling pipeline for intertemporal choice (SS vs LL).
Runs three models: Logistic Regression, Random Forest, XGBoost.
Reads clean_data_basic.csv (output of data_hackathon_analysis.py).

Usage:
    python predictive_model.py
    python predictive_model.py --input clean_data_basic.csv --sample 300000
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_style("whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT = Path("clean_data_basic.csv")
PLOT_DIR = Path("images")
SAMPLE_N = 300_000          # cap for speed; set None to use all rows
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _log(msg: str) -> None:
    print(msg, flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def load_and_prepare(path: Path, sample_n: int | None = SAMPLE_N) -> pd.DataFrame:
    """Load cleaned CSV and engineer modeling features."""
    _log(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    _log(f"  Raw shape: {df.shape}")

    # ── Filter to valid binary choice rows ────────────────────────────────
    df = df[df["choice"].isin([0, 1])].copy()

    # ── Core numeric features (already in clean data) ─────────────────────
    df["ss_value"]  = pd.to_numeric(df["ss_value"],  errors="coerce")
    df["ll_value"]  = pd.to_numeric(df["ll_value"],  errors="coerce")
    df["ss_time"]   = pd.to_numeric(df["ss_time"],   errors="coerce")
    df["ll_time"]   = pd.to_numeric(df["ll_time"],   errors="coerce")
    df["rt"]        = pd.to_numeric(df["rt"],         errors="coerce")
    df["age"]       = pd.to_numeric(df["age"],        errors="coerce")

    # ── Engineered features ───────────────────────────────────────────────
    # Reward ratio — how many times bigger is LL than SS?
    df["reward_ratio"] = np.where(
        df["ss_value"] > 0,
        df["ll_value"] / df["ss_value"],
        np.nan,
    )

    # Value difference (absolute gain for waiting)
    df["value_diff"] = df["ll_value"] - df["ss_value"]

    # Time difference in days (cost of waiting)
    df["time_diff_days"] = df["ll_time"] - df["ss_time"]

    # Daily gain rate — value gained per day of waiting
    df["daily_gain"] = np.where(
        df["time_diff_days"] > 0,
        df["value_diff"] / df["time_diff_days"],
        np.nan,
    )

    # Relative delay — LL delay as fraction of total delay span
    total_delay = df["ll_time"] + df["ss_time"]
    df["relative_delay"] = np.where(
        total_delay > 0,
        df["ll_time"] / total_delay,
        np.nan,
    )

    # Log response time (skewed distribution → log helps linear models)
    df["log_rt"] = np.where(df["rt"] > 0, np.log1p(df["rt"]), np.nan)

    # ── Encode categoricals ───────────────────────────────────────────────
    cat_cols = ["procedure", "incentivization", "presentation_of_information", "fixed_attributes"]
    label_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "missing", "": "missing"})
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col])
            label_encoders[col] = le

    # ── Binary flags ──────────────────────────────────────────────────────
    for col in ["online_study", "time_pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── Select final feature set ──────────────────────────────────────────
    feature_cols = [
        # Task-level
        "ss_value", "ll_value", "ss_time", "ll_time",
        "reward_ratio", "value_diff", "time_diff_days",
        "daily_gain", "relative_delay",
        # Behavioral
        "log_rt",
        # Demographics
        "age",
        # Context
        "online_study", "time_pressure",
        "procedure_enc", "incentivization_enc",
        "presentation_of_information_enc", "fixed_attributes_enc",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # ── Drop rows with NaN in any feature ─────────────────────────────────
    model_df = df[feature_cols + ["choice"]].dropna()
    _log(f"  Rows available for modeling: {len(model_df):,}")

    # ── Subsample for speed ───────────────────────────────────────────────
    if sample_n and len(model_df) > sample_n:
        model_df = model_df.sample(n=sample_n, random_state=RANDOM_STATE)
        _log(f"  Subsampled to: {len(model_df):,}")

    return model_df, feature_cols, label_encoders


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train three models, evaluate, and return results dict."""

    X = df[feature_cols].values
    y = df["choice"].astype(int).values

    # ── Train / Test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    _log(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    _log(f"  Class balance (train): SS={np.mean(y_train == 0):.1%}, LL={np.mean(y_train == 1):.1%}")

    # ── Scale features (needed for Logistic Regression) ───────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}

    # ── Model 1: Logistic Regression ──────────────────────────────────────
    _log("\n── Logistic Regression ──")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

    acc_lr  = accuracy_score(y_test, y_pred_lr)
    auc_lr  = roc_auc_score(y_test, y_prob_lr)
    _log(f"  Accuracy: {acc_lr:.4f}")
    _log(f"  AUC-ROC:  {auc_lr:.4f}")
    _log(f"\n{classification_report(y_test, y_pred_lr, target_names=['SS (0)', 'LL (1)'])}")

    results["Logistic Regression"] = {
        "model": lr, "scaler": scaler,
        "y_pred": y_pred_lr, "y_prob": y_prob_lr,
        "accuracy": acc_lr, "auc": auc_lr,
    }

    # ── Model 2: Random Forest ────────────────────────────────────────────
    _log("── Random Forest ──")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=20,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    _log(f"  Accuracy: {acc_rf:.4f}")
    _log(f"  AUC-ROC:  {auc_rf:.4f}")
    _log(f"\n{classification_report(y_test, y_pred_rf, target_names=['SS (0)', 'LL (1)'])}")

    results["Random Forest"] = {
        "model": rf,
        "y_pred": y_pred_rf, "y_prob": y_prob_rf,
        "accuracy": acc_rf, "auc": auc_rf,
    }

    # ── Model 3: XGBoost ─────────────────────────────────────────────────
    _log("── XGBoost ──")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        _log(f"  Accuracy: {acc_xgb:.4f}")
        _log(f"  AUC-ROC:  {auc_xgb:.4f}")
        _log(f"\n{classification_report(y_test, y_pred_xgb, target_names=['SS (0)', 'LL (1)'])}")

        results["XGBoost"] = {
            "model": xgb,
            "y_pred": y_pred_xgb, "y_prob": y_prob_xgb,
            "accuracy": acc_xgb, "auc": auc_xgb,
        }
    except ImportError:
        _log("  ⚠ xgboost not installed — run: pip install xgboost")
        _log("  Skipping XGBoost.")

    return results, X_test, X_test_sc, y_test, feature_cols, scaler


# ═════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(results: dict, save_dir: Path) -> None:
    """Bar chart comparing accuracy and AUC across models."""
    save_dir.mkdir(exist_ok=True)

    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    aucs  = [results[n]["auc"]      for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    bars1 = axes[0].bar(names, accs, color=["#2A9D8F", "#E76F51", "#5B8DEF"][:len(names)], edgecolor="white")
    axes[0].set_ylim(0.5, max(accs) + 0.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy Comparison", fontweight="bold")
    for bar, val in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.2%}", ha="center", fontweight="bold", fontsize=11)

    # AUC
    bars2 = axes[1].bar(names, aucs, color=["#2A9D8F", "#E76F51", "#5B8DEF"][:len(names)], edgecolor="white")
    axes[1].set_ylim(0.5, max(aucs) + 0.05)
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Model AUC-ROC Comparison", fontweight="bold")
    for bar, val in zip(bars2, aucs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_dir / "plot_model_comparison.png", bbox_inches="tight", dpi=150)
    plt.show()
    _log(f"  Saved: {save_dir / 'plot_model_comparison.png'}")


def plot_roc_curves(results: dict, y_test: np.ndarray, save_dir: Path) -> None:
    """Overlay ROC curves for all models."""
    save_dir.mkdir(exist_ok=True)
    colors = {"Logistic Regression": "#2A9D8F", "Random Forest": "#E76F51", "XGBoost": "#5B8DEF"}

    plt.figure(figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.4f})',
                 color=colors.get(name, "gray"), linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — SS vs LL Choice Prediction", fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / "plot_roc_curves.png", bbox_inches="tight", dpi=150)
    plt.show()
    _log(f"  Saved: {save_dir / 'plot_roc_curves.png'}")


def plot_confusion_matrices(results: dict, y_test: np.ndarray, save_dir: Path) -> None:
    """Side-by-side confusion matrices."""
    save_dir.mkdir(exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["SS (0)", "LL (1)"],
                    yticklabels=["SS (0)", "LL (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nAcc={res['accuracy']:.2%}", fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "plot_confusion_matrices.png", bbox_inches="tight", dpi=150)
    plt.show()
    _log(f"  Saved: {save_dir / 'plot_confusion_matrices.png'}")


def plot_feature_importance(results: dict, feature_cols: list[str],
                            scaler, save_dir: Path) -> None:
    """Feature importance from LR coefficients and RF/XGB importances."""
    save_dir.mkdir(exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, max(6, len(feature_cols) * 0.35)))

    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        model = res["model"]

        if name == "Logistic Regression":
            # Use absolute coefficient magnitude (model was trained on scaled data)
            importances = np.abs(model.coef_[0])
            label = "| Coefficient | (scaled features)"
        else:
            importances = model.feature_importances_
            label = "Feature importance"

        order = np.argsort(importances)
        ax.barh(
            [feature_cols[i] for i in order],
            importances[order],
            color="#5B8DEF", edgecolor="white",
        )
        ax.set_xlabel(label)
        ax.set_title(f"{name}", fontweight="bold")

    plt.suptitle("What Drives SS vs LL Choice?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "plot_feature_importance.png", bbox_inches="tight", dpi=150)
    plt.show()
    _log(f"  Saved: {save_dir / 'plot_feature_importance.png'}")


def plot_lr_coefficients(lr_model, feature_cols: list[str], save_dir: Path) -> None:
    """Signed LR coefficients — interpretable direction of effect."""
    save_dir.mkdir(exist_ok=True)

    coefs = lr_model.coef_[0]
    order = np.argsort(coefs)

    colors = ["#E76F51" if c < 0 else "#2A9D8F" for c in coefs[order]]

    plt.figure(figsize=(9, max(5, len(feature_cols) * 0.38)))
    plt.barh([feature_cols[i] for i in order], coefs[order], color=colors, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (positive = favors LL / patience)")
    plt.title("Logistic Regression Coefficients\n(Scaled Features — Direction Matters)",
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "plot_lr_coefficients.png", bbox_inches="tight", dpi=150)
    plt.show()
    _log(f"  Saved: {save_dir / 'plot_lr_coefficients.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. BUSINESS FRAMING — STREAMING SUBSCRIPTION APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
def print_business_framing(results: dict, feature_cols: list[str]) -> None:
    """Print the business application narrative connecting findings to
    streaming platform subscription tier optimization."""

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_auc  = results[best_name]["auc"]
    best_acc  = results[best_name]["accuracy"]

    # Get LR coefficients for interpretable insights
    lr = results.get("Logistic Regression")
    if lr:
        coefs = dict(zip(feature_cols, lr["model"].coef_[0]))

    _log("\n" + "=" * 72)
    _log("  BUSINESS FRAMING: STREAMING SUBSCRIPTION TIER OPTIMIZATION")
    _log("=" * 72)

    _log("""
CONTEXT
-------
Digital streaming platforms offer tiered subscription models — e.g.,
free/ad-supported, standard, and premium ad-free. The strategic question
is: when will a user choose the "larger-later" premium tier (higher cost
but better long-term experience) over the "smaller-sooner" free tier
(instant access, but ad-interrupted)?

Our intertemporal choice models directly map to this decision:
  SS (0) → Free / low tier  (immediate gratification, lower value)
  LL (1) → Premium tier     (delayed payoff, higher value)

HOW OUR MODELS APPLY
---------------------""")

    _log(f"""
1. PREDICTING TIER CONVERSION
   Our best model ({best_name}) predicts SS-vs-LL choice with
   {best_acc:.1%} accuracy and {best_auc:.4f} AUC-ROC. Applied to a
   streaming platform, this means we can predict which users are likely
   to convert to premium based on how they respond to value-vs-delay
   trade-offs in their behavior patterns.

2. THE 1.29× TIPPING POINT — PRICING STRATEGY
   From our EDA, users cross the 50% LL threshold when the reward ratio
   reaches ~1.29×. For a streaming service, this means:
   → If the free tier is valued at ~$10/month in perceived utility,
     the premium tier must deliver at least ~$12.90 in perceived value
     for most users to switch.
   → Pricing the premium tier too close to free-tier utility
     (ratio < 1.29×) will result in low conversion.

3. FEATURE-DRIVEN SEGMENTATION""")

    if lr:
        _log(f"""
   Our logistic regression reveals which factors push users toward
   patience (premium) or impatience (free tier):
   → reward_ratio coefficient: {coefs.get('reward_ratio', 0):+.3f}
     Higher perceived value gap → more premium adoption
   → time_diff_days coefficient: {coefs.get('time_diff_days', 0):+.3f}
     Longer wait to realize value → less premium adoption
   → online_study coefficient: {coefs.get('online_study', 0):+.3f}
     Digital-native users may behave differently than in-store
   → age coefficient: {coefs.get('age', 0):+.3f}
     Age-related patience patterns inform targeting

   ACTIONABLE INSIGHT: Focus marketing spend on users whose behavioral
   profile predicts LL tendency — they are most likely to convert and
   retain on premium tiers.""")

    _log("""
4. CONTEXT-DEPENDENT ENGAGEMENT
   Our analysis shows task format (procedure) explains a ~30pp spread
   in patience. For streaming platforms:
   → HOW you present the upgrade offer matters more than WHO you
     show it to.
   → A/B test the upgrade prompt format — framing the premium tier
     as "unlocking value" vs "avoiding ads" may shift conversion
     by double digits.
   → Online channels trend ~8pp more impatient — mobile push
     notifications may need a stronger value pitch than in-app
     prompts shown during engagement.

5. CUSTOMER LIFETIME VALUE (CLV) IMPLICATIONS
   Users who exhibit "patient" decision profiles (predicted LL):
   → Higher expected retention (willing to invest for long-term value)
   → Lower churn risk for premium tiers
   → Better candidates for annual plan upsells
   
   Users with "impatient" profiles (predicted SS):
   → Target with trial offers and short-commitment plans
   → Emphasize immediate benefits ("start watching ad-free tonight")
   → Monitor for churn triggers and intervene with retention offers

RECOMMENDATIONS
---------------
  a) Price premium tiers at ≥1.29× the perceived value of free tier
  b) A/B test upgrade prompt FORMAT before segmenting by demographics
  c) Use model predictions to personalize upgrade timing and messaging
  d) Offer shorter commitment periods to high-SS-probability segments
  e) Track reward_ratio sensitivity by cohort to detect market shifts
""")

    _log("=" * 72)


# ═════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
def print_summary_table(results: dict) -> None:
    """Print a clean comparison table."""
    _log("\n┌──────────────────────┬────────────┬────────────┐")
    _log("│ Model                │  Accuracy  │  AUC-ROC   │")
    _log("├──────────────────────┼────────────┼────────────┤")
    for name, res in results.items():
        _log(f"│ {name:<20} │  {res['accuracy']:.4f}    │  {res['auc']:.4f}    │")
    _log("└──────────────────────┴────────────┴────────────┘")

    best = max(results, key=lambda k: results[k]["auc"])
    _log(f"\n  ★ Best model by AUC: {best} ({results[best]['auc']:.4f})")
    _log(f"    Methodological note: Logistic Regression is preferred for")
    _log(f"    interpretability; {best} is preferred for raw performance.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive modeling for intertemporal choice")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample", type=int,  default=SAMPLE_N)
    args = parser.parse_args([])  # Empty list for Colab compatibility

    # ── Load & Prepare ────────────────────────────────────────────────────
    df, feature_cols, label_encoders = load_and_prepare(args.input, args.sample)

    # ── Train & Evaluate ──────────────────────────────────────────────────
    _log("\n" + "=" * 50)
    _log("  TRAINING MODELS")
    _log("=" * 50)
    results, X_test, X_test_sc, y_test, feature_cols, scaler = train_and_evaluate(df, feature_cols)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary_table(results)

    # ── Plots ─────────────────────────────────────────────────────────────
    _log("\n" + "=" * 50)
    _log("  GENERATING VISUALIZATIONS")
    _log("=" * 50)
    plot_model_comparison(results, PLOT_DIR)
    plot_roc_curves(results, y_test, PLOT_DIR)
    plot_confusion_matrices(results, y_test, PLOT_DIR)
    plot_feature_importance(results, feature_cols, scaler, PLOT_DIR)

    if "Logistic Regression" in results:
        plot_lr_coefficients(results["Logistic Regression"]["model"], feature_cols, PLOT_DIR)

    # ── Business Framing ──────────────────────────────────────────────────
    print_business_framing(results, feature_cols)

    _log("\n✅ Done. All plots saved to ./images/")
    _log("   Next: push to GitHub and update your GitHub Pages site.\n")


if __name__ == "__main__":
    main()
