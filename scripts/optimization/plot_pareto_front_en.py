"""Plot the English Figure 7 Pareto front for the manuscript.

Inputs:
    experiments/3000_fixed_temp/optimization/optimization_results.csv
    experiments/3000_fixed_temp/optimization/optimization_representative_solutions.csv

Outputs:
    experiments/3000_fixed_temp/optimization/fig7_pareto_front_en.png
    experiments/3000_fixed_temp/optimization/fig7_pareto_front_en.pdf
    experiments/3000_fixed_temp/optimization/fig7_pareto_front_en.svg

This script only regenerates the figure and does not modify optimization CSVs.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OPT_DIR = PROJECT_ROOT / "experiments" / "3000_fixed_temp" / "optimization"

PARETO_CSV = OPT_DIR / "optimization_results.csv"
REPRESENTATIVE_CSV = OPT_DIR / "optimization_representative_solutions.csv"
OUTPUT_STEM = OPT_DIR / "fig7_pareto_front_en"

MATTE_CU_TARGET = 60.0
MATTE_CU_BAND = (59.9, 60.1)
FESIO2_LO = 1.15
FESIO2_HI = 1.30


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pareto = pd.read_csv(PARETO_CSV)
    reps = pd.read_csv(REPRESENTATIVE_CSV).sort_values(
        "slag_Cu_pct"
    ).reset_index(drop=True)
    required = {"slag_Cu_pct", "matte_Cu_pct", "slag_FeO_SiO2_ratio", "solution_idx"}
    for path, data in ((PARETO_CSV, pareto), (REPRESENTATIVE_CSV, reps)):
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    # 与优化主程序一致，使用按渣含 Cu 排序后的中间代表方案（方案 C）。
    recommended = reps.iloc[len(reps) // 2]
    return pareto, reps, recommended


def plot(pareto: pd.DataFrame, reps: pd.DataFrame, recommended: pd.Series) -> None:
    configure_style()

    feasible = pareto[
        pareto["slag_FeO_SiO2_ratio"].between(FESIO2_LO, FESIO2_HI, inclusive="both")
    ].copy()
    if feasible.empty:
        raise ValueError("No feasible Pareto points within the FeO/SiO2 range.")

    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    ax.axhspan(MATTE_CU_BAND[0], MATTE_CU_BAND[1], color="#d4544a", alpha=0.08, zorder=0)
    ax.axhline(
        MATTE_CU_TARGET,
        color="#d4544a",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
        label="Matte Cu target = 60 wt%",
        zorder=1,
    )

    scatter = ax.scatter(
        feasible["slag_Cu_pct"],
        feasible["matte_Cu_pct"],
        c=feasible["slag_FeO_SiO2_ratio"],
        cmap="viridis",
        s=42,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
        rasterized=True,
        zorder=2,
    )

    labels = ["A", "B", "C", "D", "E"]
    offsets = [(8, 8), (9, 8), (8, 7), (7, -18), (8, 8)]
    for label, offset, (_, row) in zip(labels, offsets, reps.iterrows()):
        ax.annotate(
            label,
            xy=(row["slag_Cu_pct"], row["matte_Cu_pct"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#d4544a",
            zorder=4,
        )

    ax.scatter(
        recommended["slag_Cu_pct"],
        recommended["matte_Cu_pct"],
        marker="*",
        s=270,
        color="#e4584d",
        edgecolor="black",
        linewidth=0.8,
        label="Recommended solution",
        zorder=5,
    )

    ax.set_xlabel("Slag Cu / wt%", fontsize=10)
    ax.set_ylabel("Matte Cu / wt%", fontsize=10)
    ax.set_title("NSGA-II Pareto Front for Intelligent Blending", fontsize=11.5)
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.75)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)

    x_min = feasible["slag_Cu_pct"].min()
    x_max = feasible["slag_Cu_pct"].max()
    y_min = feasible["matte_Cu_pct"].min()
    y_max = feasible["matte_Cu_pct"].max()
    ax.set_xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.05)
    ax.set_ylim(min(MATTE_CU_BAND[0], y_min) - 0.02, max(MATTE_CU_BAND[1], y_max) + 0.02)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label("Slag FeO/SiO$_2$", fontsize=9.5)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pareto, reps, recommended = load_data()
    plot(pareto, reps, recommended)
    print(f"Pareto points: {len(pareto)}")
    print(
        "Feasible points: "
        f"{pareto['slag_FeO_SiO2_ratio'].between(FESIO2_LO, FESIO2_HI, inclusive='both').sum()}"
    )
    print(
        "Recommended solution: "
        f"solution_idx={int(recommended['solution_idx'])}, "
        f"slag_Cu={recommended['slag_Cu_pct']:.4f} wt%, "
        f"matte_Cu={recommended['matte_Cu_pct']:.2f} wt%, "
        f"FeO/SiO2={recommended['slag_FeO_SiO2_ratio']:.3f}"
    )
    print(f"Saved: {OUTPUT_STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
