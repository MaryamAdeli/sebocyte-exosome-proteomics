"""
Sebocyte exosome proteomics batch-correction workflow
=====================================================

This script processes merged LFQ proteomics data from two LC-MS/MS platforms
(Astral and Exploris) for control and MSC-derived exosome-treated sebocytes.

Workflow:
1. Load a Perseus-merged MaxQuant LFQ intensity table.
2. Define platform/batch and biological condition metadata.
3. Plot LFQ intensity distributions before batch correction.
4. Apply ComBat batch correction using LC-MS/MS platform as the batch variable
   and treatment condition as a categorical biological covariate.
5. Restore the original missing-value structure after ComBat correction.
6. Filter proteins based on valid values in control or exosome groups.
7. Perform Welch's t-test, Benjamini-Hochberg FDR correction, and log2FC calculation.
8. Save corrected data, volcano statistics, and before/after violin plots.

Input file:
- Expected to be an Excel file exported/merged from Perseus after standard filtering.
- LFQ intensity values are expected to be log2-transformed. If your input is not
  log2-transformed, set ALREADY_LOG2 = False below.

Author: Maryam Adelipour
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

try:
    from neuroCombat import neuroCombat
except ImportError as exc:
    raise ImportError(
        "The neuroCombat package is required. Install it with: pip install neuroCombat"
    ) from exc


# =============================================================================
# User settings
# =============================================================================

# Use relative paths for GitHub/reproducibility.
BASE_DIR = Path(".")
INPUT_FILE = BASE_DIR / "data" / "Merged_Exosome_Data.xlsx"

RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

# If the LFQ values were already log2-transformed in Perseus, keep this True.
# If the values are raw MaxQuant LFQ intensities, set this to False.
ALREADY_LOG2 = True

# Protein/gene annotation column from the Perseus table.
GENE_COLUMN = "T: Gene names"

# LFQ intensity columns from the Astral dataset.
ASTRAL_COLS = [
    "LFQ intensity Control_1",
    "LFQ intensity Control_2",
    "LFQ intensity Control_3",
    "LFQ intensity Exo_1",
    "LFQ intensity Exo_2",
    "LFQ intensity Exo_3",
]

# LFQ intensity columns from the Exploris dataset.
EXPLORIS_COLS = [
    "LFQ intensity Control 1",
    "LFQ intensity Control 2",
    "LFQ intensity Control 3",
    "LFQ intensity Exo5_1",
    "LFQ intensity Exo5_2",
    "LFQ intensity Exo5_3",
]

# Filtering and significance thresholds.
MIN_VALID_PER_PLATFORM = 4  # for initial ComBat input filtering
MIN_VALID_PER_GROUP = 4     # for downstream statistical testing
FDR_THRESHOLD = 0.10
LOG2FC_THRESHOLD = 0.10     # use 1.0 for a two-fold cutoff; 0.58 for 1.5-fold


# =============================================================================
# Helper functions
# =============================================================================

def make_output_dirs() -> None:
    """Create output folders if they do not exist."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def check_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            "The following required columns were not found in the input file:\n"
            + "\n".join(missing)
        )


def load_input_table(input_file: Path) -> pd.DataFrame:
    """Load the Perseus-merged Excel file."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Place the input file in the data/ folder or update INPUT_FILE."
        )

    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()
    return df


def build_metadata(expr_cols: list[str]) -> pd.DataFrame:
    """Build sample metadata for ComBat correction."""
    batch = ["Astral"] * len(ASTRAL_COLS) + ["Exploris"] * len(EXPLORIS_COLS)
    condition = [
        "Control", "Control", "Control", "Exo", "Exo", "Exo",
        "Control", "Control", "Control", "Exo", "Exo", "Exo",
    ]

    meta = pd.DataFrame(
        {"sample": expr_cols, "batch": batch, "condition": condition}
    ).set_index("sample")
    return meta


def prepare_expression_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract LFQ intensity columns and build metadata."""
    expr_cols = ASTRAL_COLS + EXPLORIS_COLS
    check_required_columns(df, expr_cols)

    expr = df[expr_cols].apply(pd.to_numeric, errors="coerce")

    if not ALREADY_LOG2:
        # MaxQuant missing LFQ values are often zeros; convert to missing before log2.
        expr = expr.replace(0, np.nan)
        expr = np.log2(expr)

    meta = build_metadata(expr_cols)
    return expr, meta


def filter_for_combat(expr: pd.DataFrame) -> pd.Series:
    """
    Keep proteins with sufficient valid values in at least one platform-specific dataset.
    This reduces very sparse proteins before ComBat correction.
    """
    valid_mask = (
        (expr[ASTRAL_COLS].notna().sum(axis=1) >= MIN_VALID_PER_PLATFORM)
        | (expr[EXPLORIS_COLS].notna().sum(axis=1) >= MIN_VALID_PER_PLATFORM)
    )
    return valid_mask


def run_combat_batch_correction(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ComBat correction using platform as batch variable.

    Missing values are temporarily replaced with column-wise mean values because
    ComBat requires a complete matrix. After correction, the original missing-value
    structure is restored.
    """
    valid_mask = filter_for_combat(expr)
    expr_valid = expr.loc[valid_mask].copy()

    # Temporary column-wise mean imputation only for ComBat computation.
    expr_imputed = expr_valid.fillna(expr_valid.mean(axis=0))

    covars_df = meta.loc[expr_imputed.columns, ["batch", "condition"]].copy()
    assert covars_df.index.equals(expr_imputed.columns)

    combat_out = neuroCombat(
        dat=expr_imputed.values,
        covars=covars_df,
        batch_col="batch",
        categorical_cols=["condition"],
        continuous_cols=None,
    )

    expr_corrected = pd.DataFrame(
        combat_out["data"],
        index=expr_valid.index,
        columns=expr_valid.columns,
    )

    # Restore original missing values.
    expr_corrected[expr_valid.isna()] = np.nan

    # Return full-size matrix with corrected values only for valid proteins.
    expr_final = expr.copy()
    expr_final.loc[valid_mask, expr.columns] = expr_corrected
    return expr_final


def make_plot_dataframe(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Convert expression matrix to long format for plotting."""
    plot_df = expr.melt(var_name="sample", value_name="intensity")
    plot_df["batch"] = plot_df["sample"].map(meta["batch"])
    plot_df["condition"] = plot_df["sample"].map(meta["condition"])
    return plot_df


def plot_sample_violins(
    expr: pd.DataFrame,
    title: str,
    output_prefix: str,
    ylabel: str = "log2 LFQ intensity",
) -> None:
    """Plot sample-wise violin plots for LFQ intensity distributions."""
    sample_order = ASTRAL_COLS + EXPLORIS_COLS
    data = [expr[s].dropna().values for s in sample_order]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.violinplot(data, showmeans=True, showextrema=False)

    ax.set_xticks(range(1, len(sample_order) + 1))
    ax.set_xticklabels(sample_order, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.axvline(x=len(ASTRAL_COLS) + 0.5, linestyle="--", linewidth=1, color="gray")

    # Add simple platform labels.
    y_top = ax.get_ylim()[1]
    ax.text(3.5, y_top, "Astral", ha="center", va="bottom", fontsize=9)
    ax.text(9.5, y_top, "Exploris", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    png_path = FIGURE_DIR / f"{output_prefix}.png"
    pdf_path = FIGURE_DIR / f"{output_prefix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_batch_violins(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    title: str,
    output_prefix: str,
    ylabel: str = "log2 LFQ intensity",
) -> None:
    """Plot batch-level violin plots for Astral vs Exploris."""
    plot_df = make_plot_dataframe(expr, meta)
    batches = ["Astral", "Exploris"]
    data = [plot_df.loc[plot_df["batch"] == b, "intensity"].dropna().values for b in batches]

    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.violinplot(data, showmeans=True, showextrema=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(batches)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    png_path = FIGURE_DIR / f"{output_prefix}.png"
    pdf_path = FIGURE_DIR / f"{output_prefix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def calculate_volcano_statistics(df: pd.DataFrame, expr_corrected: pd.DataFrame) -> pd.DataFrame:
    """Calculate log2FC, Welch's t-test p-values, FDR, and regulation category."""
    control_cols = [c for c in expr_corrected.columns if "Control" in c]
    exo_cols = [c for c in expr_corrected.columns if "Exo" in c]

    group_valid_mask = (
        (expr_corrected[control_cols].notna().sum(axis=1) >= MIN_VALID_PER_GROUP)
        | (expr_corrected[exo_cols].notna().sum(axis=1) >= MIN_VALID_PER_GROUP)
    )

    expr_valid = expr_corrected.loc[group_valid_mask].copy()

    if GENE_COLUMN in df.columns:
        gene_names = df.loc[group_valid_mask, GENE_COLUMN].astype(str)
    else:
        warnings.warn(
            f"Gene column '{GENE_COLUMN}' not found. Using row index as gene_name."
        )
        gene_names = pd.Series(expr_valid.index.astype(str), index=expr_valid.index)

    rows = []
    for idx, row in expr_valid.iterrows():
        ctrl = row[control_cols].dropna().astype(float).values
        exo = row[exo_cols].dropna().astype(float).values

        if len(ctrl) >= 2 and len(exo) >= 2:
            log2fc = float(np.mean(exo) - np.mean(ctrl))
            _, p_value = ttest_ind(exo, ctrl, equal_var=False)
        else:
            log2fc = np.nan
            p_value = np.nan

        rows.append(
            {
                "row_index": idx,
                "gene_name": gene_names.loc[idx],
                "log2FC": log2fc,
                "p_value": p_value,
                "n_control": len(ctrl),
                "n_exo": len(exo),
            }
        )

    stats = pd.DataFrame(rows)

    valid_p = stats["p_value"].notna()
    stats["FDR"] = np.nan
    if valid_p.any():
        stats.loc[valid_p, "FDR"] = multipletests(
            stats.loc[valid_p, "p_value"], method="fdr_bh"
        )[1]

    eps = 1e-300
    stats["neg_log10_p"] = -np.log10(np.clip(stats["p_value"].fillna(1.0), eps, 1.0))
    stats["neg_log10_FDR"] = -np.log10(np.clip(stats["FDR"].fillna(1.0), eps, 1.0))

    stats["significant_FDR"] = (
        (stats["FDR"] < FDR_THRESHOLD)
        & (np.abs(stats["log2FC"]) >= LOG2FC_THRESHOLD)
    )

    stats["regulation"] = "NS"
    stats.loc[stats["significant_FDR"] & (stats["log2FC"] > 0), "regulation"] = "Up"
    stats.loc[stats["significant_FDR"] & (stats["log2FC"] < 0), "regulation"] = "Down"

    return stats


def save_corrected_table(df: pd.DataFrame, expr_corrected: pd.DataFrame) -> pd.DataFrame:
    """Save a copy of the original table with corrected LFQ columns."""
    corrected_df = df.copy()
    corrected_df.loc[:, expr_corrected.columns] = expr_corrected

    corrected_path = TABLE_DIR / "exosome_proteomics_combat_corrected_matrix.xlsx"
    corrected_df.to_excel(corrected_path, index=False)

    return corrected_df


def plot_volcano(stats: pd.DataFrame, output_prefix: str = "exosome_vs_control_volcano") -> None:
    """Generate a volcano plot from the statistics table."""
    color_map = {"NS": "lightgray", "Up": "#D55E00", "Down": "#0072B2"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for category in ["NS", "Down", "Up"]:
        subset = stats[stats["regulation"] == category]
        ax.scatter(
            subset["log2FC"],
            subset["neg_log10_FDR"],
            s=25 if category == "NS" else 35,
            c=color_map[category],
            label=category,
            alpha=0.75,
            edgecolors="none",
        )

    ax.axvline(LOG2FC_THRESHOLD, linestyle="--", linewidth=1, color="gray")
    ax.axvline(-LOG2FC_THRESHOLD, linestyle="--", linewidth=1, color="gray")
    ax.axhline(-np.log10(FDR_THRESHOLD), linestyle="--", linewidth=1, color="gray")

    ax.set_xlabel("log2 fold change (Exo vs Control)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("MSC-derived exosome-treated vs control sebocytes")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(FIGURE_DIR / f"{output_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    make_output_dirs()

    df = load_input_table(INPUT_FILE)
    expr, meta = prepare_expression_matrix(df)

    print(f"Loaded input table: {df.shape[0]} proteins × {df.shape[1]} columns")
    print(f"Expression matrix: {expr.shape[0]} proteins × {expr.shape[1]} samples")

    # Pre-correction QC plots.
    plot_batch_violins(
        expr,
        meta,
        title="Before batch correction",
        output_prefix="violin_by_batch_before_combat",
    )
    plot_sample_violins(
        expr,
        title="Before batch correction",
        output_prefix="violin_by_sample_before_combat",
    )

    # Batch correction.
    expr_corrected = run_combat_batch_correction(expr, meta)

    # Post-correction QC plots.
    plot_batch_violins(
        expr_corrected,
        meta,
        title="After ComBat batch correction",
        output_prefix="violin_by_batch_after_combat",
    )
    plot_sample_violins(
        expr_corrected,
        title="After ComBat batch correction",
        output_prefix="violin_by_sample_after_combat",
    )

    # Save corrected table.
    corrected_df = save_corrected_table(df, expr_corrected)
    print("Saved corrected matrix to results/tables/exosome_proteomics_combat_corrected_matrix.xlsx")

    # Statistics.
    stats = calculate_volcano_statistics(corrected_df, expr_corrected)
    stats_path_xlsx = TABLE_DIR / "exosome_vs_control_volcano_statistics.xlsx"
    stats_path_csv = TABLE_DIR / "exosome_vs_control_volcano_statistics.csv"
    stats.to_excel(stats_path_xlsx, index=False)
    stats.to_csv(stats_path_csv, index=False)

    print("Saved volcano statistics:")
    print(f"  {stats_path_xlsx}")
    print(f"  {stats_path_csv}")
    print(stats["regulation"].value_counts(dropna=False))

    # Volcano plot.
    plot_volcano(stats)
    print("Saved volcano plot to results/figures/")


if __name__ == "__main__":
    main()
