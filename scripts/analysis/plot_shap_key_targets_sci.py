"""
Publication-style SHAP analysis for key metallurgical targets.

The script recomputes first-order SHAP values for the fitted TabPFN models of
four key targets and exports both numeric importance tables and SCI-style
figures.

Usage:
    conda run -n tabpfn python scripts/analysis/plot_shap_key_targets_sci.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "3000_fixed_temp" / "ml_dataset.csv"
MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "3000_fixed_temp"
    / "tabpfn"
    / "20260622_2247"
    / "saved_models"
)
OUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "3000_fixed_temp"
    / "tabpfn"
    / "20260622_2247"
    / "shap_results_sci_key_targets"
)
IMPORTANCE_CSV = OUT_DIR / "shap_key_targets_importance.csv"
TABPFN_MODEL_PATH = os.environ.get("TABPFN_MODEL_PATH", "auto")  # 'auto' = 自动下载

DEVICE = os.environ.get("TABPFN_DEVICE", "cuda")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BACKGROUND = 200
N_EXPLAIN = 50
BUDGET = 256

INPUT_COLS = [
    "T(℃)",
    "P(atm)",
    "Cu(g)",
    "S(g)",
    "Fe(g)",
    "SiO2(g)",
    "Pb(g)",
    "Zn(g)",
    "MgO(g)",
    "Al2O3(g)",
    "Ni(g)",
    "Sb(g)",
    "Bi(g)",
    "CaO(g)",
    "As(g)",
    "O2(g)",
]

# The manuscript heatmap focuses on the 13 charge-composition inputs. The
# fixed operating conditions (temperature, pressure, and oxygen) are omitted.
HEATMAP_INPUT_COLS = [
    "Cu(g)",
    "S(g)",
    "Fe(g)",
    "SiO2(g)",
    "Pb(g)",
    "Zn(g)",
    "MgO(g)",
    "Al2O3(g)",
    "Ni(g)",
    "Sb(g)",
    "Bi(g)",
    "CaO(g)",
    "As(g)",
]

FEATURE_LABELS = {
    "T(℃)": "T",
    "P(atm)": "P",
    "Cu(g)": "Cu",
    "S(g)": "S",
    "Fe(g)": "Fe",
    "SiO2(g)": "SiO$_2$",
    "Pb(g)": "Pb",
    "Zn(g)": "Zn",
    "MgO(g)": "MgO",
    "Al2O3(g)": "Al$_2$O$_3$",
    "Ni(g)": "Ni",
    "Sb(g)": "Sb",
    "Bi(g)": "Bi",
    "CaO(g)": "CaO",
    "As(g)": "As",
    "O2(g)": "O$_2$",
}

KEY_TARGETS = [
    "matte_Cu_pct",
    "slag_Cu_pct",
    "slag_FeO_pct",
    "slag_SiO2_pct",
]

TARGET_LABELS = {
    "matte_Cu_pct": "Matte Cu",
    "slag_Cu_pct": "Slag Cu",
    "slag_FeO_pct": "Slag FeO",
    "slag_SiO2_pct": "Slag SiO$_2$",
}


def get_device() -> str:
    if DEVICE != "cuda":
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_tabpfn_model(target: str, device: str):
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import InferenceEngine, _extract_archive

    model_file = MODEL_DIR / f"{target}.tabpfn_fit"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model: {model_file}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _extract_archive(model_file, tmp)

        with (tmp / "init_params.json").open(encoding="utf-8") as f:
            params = json.load(f)
        params.pop("__class_name__", None)
        params["model_path"] = str(TABPFN_MODEL_PATH)
        params["device"] = device

        est = TabPFNRegressor(**params)
        est._initialize_model_variables()
        fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
        for key, value in fitted_attrs.items():
            setattr(est, key, value)
        est.executor_ = InferenceEngine.load_state(
            tmp / "executor_state.joblib", est.models_
        )
        est.to(device)

    return est


def apply_sci_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def compute_shap_importance(target: str, model, x_background, x_explain):
    from tabpfn_extensions.interpretability import (
        shapiq as tabpfn_shapiq,
        shapiq_to_shap_explanation,
    )

    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=model,
        data=x_background,
        index="SV",
        max_order=1,
    )
    explanation = shapiq_to_shap_explanation(
        explainer,
        x_explain,
        budget=BUDGET,
        feature_names=INPUT_COLS,
    )
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)
    print(f"  {target}: computed mean |SHAP|")
    return mean_abs


def plot_heatmap(importance: pd.DataFrame) -> None:
    target_order = KEY_TARGETS
    row_max = importance.loc[target_order].max(axis=1).replace(0, 1)
    norm = importance.loc[target_order].div(row_max, axis=0)

    feature_order = (
        importance.loc[target_order, HEATMAP_INPUT_COLS]
        .mean(axis=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    norm = norm[feature_order]
    raw = importance.loc[target_order, feature_order]

    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    color_norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=1)
    im = ax.imshow(norm.values, cmap="viridis", norm=color_norm, aspect="auto")

    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in feature_order], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(target_order)))
    ax.set_yticklabels([TARGET_LABELS[t] for t in target_order])
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Prediction target")
    ax.set_title(
        "SHAP feature importance for key metallurgical targets",
        fontweight="bold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(feature_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)

    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            rgba = im.cmap(im.norm(norm.iat[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "#222222" if luminance > 0.58 else "white"
            ax.text(
                j,
                i,
                f"{raw.iat[i, j]:.2g}",
                ha="center",
                va="center",
                fontsize=5.1,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized mean |SHAP|")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)

    fig.subplots_adjust(left=0.12, right=0.94, top=0.82, bottom=0.32)
    save_figure(fig, OUT_DIR / "fig_shap_key_targets_heatmap_13_features")
    plt.close(fig)


def _plot_bar_axis(ax, importance: pd.DataFrame, target: str, n_features: int = 8) -> None:
    vals = importance.loc[target].sort_values(ascending=False).head(n_features)
    labels = [FEATURE_LABELS[c] for c in vals.index]
    y = np.arange(len(vals))[::-1]
    colors = ["#0072B2" if i == 0 else "#7AA6C2" for i in range(len(vals))]
    ax.barh(y, vals.values, color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(TARGET_LABELS[target], fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#DADADA", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    for yy, val in zip(y, vals.values):
        ax.text(val + vals.max() * 0.015, yy, f"{val:.2g}", va="center", fontsize=6.5)


def plot_target_bar(importance: pd.DataFrame, target: str) -> None:
    fig, ax = plt.subplots(figsize=(3.54, 2.75))
    _plot_bar_axis(ax, importance, target, n_features=10)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.18)
    file_label = target.replace("_pct", "").replace("_", "_").lower()
    save_figure(fig, OUT_DIR / f"fig_shap_{file_label}_bar")
    plt.close(fig)


def plot_key_target_bars(importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1))
    for ax, target in zip(axes.ravel(), KEY_TARGETS):
        _plot_bar_axis(ax, importance, target, n_features=8)
    fig.suptitle("SHAP importance for key metallurgical targets", fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.09, wspace=0.33, hspace=0.48)
    save_figure(fig, OUT_DIR / "fig_shap_key_targets_bars")
    plt.close(fig)


def load_or_compute_importance() -> pd.DataFrame:
    if IMPORTANCE_CSV.exists():
        print(f"Loading cached SHAP importance: {IMPORTANCE_CSV}")
        return pd.read_csv(IMPORTANCE_CSV, index_col=0, encoding="utf-8-sig")

    device = get_device()
    print(f"Using device: {device}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    x = df[INPUT_COLS].values.astype(np.float32)
    x_train, x_test = train_test_split(x, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    x_background = x_train[:N_BACKGROUND]
    x_explain = x_test[:N_EXPLAIN]

    rows = {}
    for target in KEY_TARGETS:
        print(f"Computing SHAP for {target}...")
        model = load_tabpfn_model(target, device)
        rows[target] = compute_shap_importance(target, model, x_background, x_explain)

    return pd.DataFrame(rows, index=INPUT_COLS).T


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_sci_style()

    importance = load_or_compute_importance()
    normalized = importance.div(importance.max(axis=1).replace(0, 1), axis=0)
    rank_rows = []
    for target in KEY_TARGETS:
        for rank, (feature, value) in enumerate(
            importance.loc[target].sort_values(ascending=False).items(), start=1
        ):
            rank_rows.append(
                {
                    "target": target,
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_shap": value,
                    "normalized_importance": normalized.loc[target, feature],
                }
            )

    importance.to_csv(OUT_DIR / "shap_key_targets_importance.csv", encoding="utf-8-sig")
    normalized.to_csv(
        OUT_DIR / "shap_key_targets_importance_normalized.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(rank_rows).to_csv(
        OUT_DIR / "shap_key_targets_ranked.csv", index=False, encoding="utf-8-sig"
    )

    plot_heatmap(importance)
    for target in KEY_TARGETS:
        plot_target_bar(importance, target)
    plot_key_target_bars(importance)
    print(f"Saved SHAP tables and figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
