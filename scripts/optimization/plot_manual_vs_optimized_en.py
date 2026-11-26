"""Plot the English Figure 8 manual-vs-optimized comparison.

Inputs:
    experiments/3000_fixed_temp/optimization/manual_blend_prediction.csv
    experiments/3000_fixed_temp/optimization/optimization_representative_solutions.csv

Outputs:
    experiments/3000_fixed_temp/optimization/fig8_manual_vs_optimized_en.png
    experiments/3000_fixed_temp/optimization/fig8_manual_vs_optimized_en.pdf
    experiments/3000_fixed_temp/optimization/fig8_manual_vs_optimized_en.svg

The optimized values use the middle representative solution (solution C),
matching the manuscript tables. This script only regenerates the figure and
does not modify optimization CSVs.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OPT_DIR = PROJECT_ROOT / "experiments" / "3000_fixed_temp" / "optimization"

MANUAL_CSV = OPT_DIR / "manual_blend_prediction.csv"
OPTIMIZED_CSV = OPT_DIR / "optimization_representative_solutions.csv"
OUTPUT_STEM = OPT_DIR / "fig8_manual_vs_optimized_en"

MATTE_CU_TARGET = 60.0
FESIO2_RANGE = (1.15, 1.30)


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


def load_values() -> tuple[pd.Series, pd.Series]:
    manual = pd.read_csv(MANUAL_CSV).iloc[0]
    optimized_all = pd.read_csv(OPTIMIZED_CSV).sort_values(
        "slag_Cu_pct"
    ).reset_index(drop=True)
    optimized = optimized_all.iloc[len(optimized_all) // 2]
    return manual, optimized


def add_value_label(ax, bar, value: float, fmt: str, pad: float) -> None:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + pad,
        fmt.format(value),
        ha="center",
        va="bottom",
        fontsize=7.6,
    )


def plot(manual: pd.Series, optimized: pd.Series) -> None:
    configure_style()
    slag_reduction = (
        (float(manual["slag_Cu_pct"]) - float(optimized["slag_Cu_pct"]))
        / float(manual["slag_Cu_pct"])
        * 100
    )

    panels = [
        {
            "key": "slag_Cu_pct",
            "ylabel": "Slag Cu / wt%",
            "ylim": (0, 0.62),
            "fmt": "{:.3f}",
            "annotation": f"{slag_reduction:.1f}% reduction",
            "annotation_color": "#d4544a",
            "annotation_y": 0.94,
        },
        {
            "key": "matte_Cu_pct",
            "ylabel": "Matte Cu / wt%",
            "ylim": (0, 65),
            "fmt": "{:.3f}",
            "target": MATTE_CU_TARGET,
            "annotation": "Target = 60 wt%",
            "annotation_color": "#d4544a",
            "annotation_y": 0.985,
        },
        {
            "key": "slag_FeO_SiO2_ratio",
            "ylabel": "Slag FeO/SiO$_2$",
            "ylim": (0, 1.75),
            "fmt": "{:.3f}",
            "target_range": FESIO2_RANGE,
            "annotation": "Target range",
            "annotation_color": "#3d7435",
            "annotation_y": 0.94,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.05), constrained_layout=True)
    colors = ["#aab8dc", "#4b4b7f"]
    labels = ["Manual\nblend", "Optimized\nblend"]

    for ax, panel in zip(axes, panels):
        key = panel["key"]
        values = [float(manual[key]), float(optimized[key])]

        if "target_range" in panel:
            lo, hi = panel["target_range"]
            ax.axhspan(lo, hi, color="#7baa72", alpha=0.13, zorder=0)
        if "target" in panel:
            ax.axhline(
                panel["target"],
                color="#d4544a",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                zorder=1,
            )

        bars = ax.bar(
            [0, 1],
            values,
            color=colors,
            edgecolor="#222222",
            linewidth=0.6,
            width=0.54,
            zorder=2,
        )
        pad = (panel["ylim"][1] - panel["ylim"][0]) * 0.015
        for bar, value in zip(bars, values):
            add_value_label(ax, bar, value, panel["fmt"], pad)

        ax.text(
            0.5,
            panel["annotation_y"],
            panel["annotation"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=panel["annotation_color"],
        )
        ax.set_ylabel(panel["ylabel"], fontsize=9.5)
        ax.set_ylim(*panel["ylim"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=7.4)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.5, alpha=0.72, zorder=0)

    fig.suptitle("Manual and Optimized Blending: Key Metallurgical Targets", fontsize=11.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    manual, optimized = load_values()
    plot(manual, optimized)
    slag_reduction = (
        (float(manual["slag_Cu_pct"]) - float(optimized["slag_Cu_pct"]))
        / float(manual["slag_Cu_pct"])
        * 100
    )
    print(
        "Manual: "
        f"slag_Cu={manual['slag_Cu_pct']:.4f}, "
        f"matte_Cu={manual['matte_Cu_pct']:.2f}, "
        f"FeO/SiO2={manual['slag_FeO_SiO2_ratio']:.3f}"
    )
    print(
        "Optimized: "
        f"solution_idx={int(optimized['solution_idx'])}, "
        f"slag_Cu={optimized['slag_Cu_pct']:.4f}, "
        f"matte_Cu={optimized['matte_Cu_pct']:.2f}, "
        f"FeO/SiO2={optimized['slag_FeO_SiO2_ratio']:.3f}"
    )
    print(f"Slag Cu reduction: {slag_reduction:.1f}%")
    print(f"Saved: {OUTPUT_STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
