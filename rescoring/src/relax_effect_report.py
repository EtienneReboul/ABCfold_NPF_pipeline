#!/usr/bin/env python3
"""
rescoring/src/relax_effect_report.py
=======================================
How much does relief.py's light, ligand-neighborhood-restricted FastRelax
actually help -- and does it help consistently across ligand categories and
backends, or are some starting from worse geometry / responding worse to
relief than others? Reads results/all_contacts.csv (aggregate.py's pooled
output) and reports raw-vs-relaxed fa_rep and total_score, broken down by
ligand and by backend.

See rescoring/README.md's "Raw (non-preminimized) poses" section for the
full-dataset headline finding this reproduces as a standing pipeline stage
(rather than the one-off analysis that originally found it): fa_rep
improves in ~99.9% of complexes (clash relief works), but total_score gets
WORSE in ~99.5% of complexes (driven by repacking/minimization strain
elsewhere in the neighborhood, not the ligand interface or fa_rep itself,
per that section's decomposition) -- this script's per-ligand/per-backend
breakdown is how you'd spot if that pattern is uneven across the dataset
(e.g. one backend's raw geometry being unusually bad, or one ligand's
poses responding worse to relief than others).

Also breaks down by ligand-POSE cluster -- the results/ligand_pose/
<protein>/pca_k3/ca_cluster_<k>/<tag>/cluster_<pose>/ sub-clusters Stage 1
(scripts/cluster_conformations.py) found within each protein's own macro-
state clusters (see data/manifest.csv's ca_cluster/ligand_pose_cluster
columns, populated by make_manifest.py from those clusters' own
assignments.parquet tables). Unlike ligand category or backend, a pose
cluster id is only meaningful WITHIN one (protein, ca_cluster) -- "pose 0"
in one protein has nothing to do with "pose 0" in another -- so this
breakdown is faceted per protein, not pooled globally, letting you spot
e.g. one particular binding pose that relieves clashes worse than the
others found for the same protein/conformation.

Outputs:
  results/relax_effect_summary.csv          per (ligand, backend): n, median/
                                              mean fa_rep raw & relaxed, % fa_rep
                                              improved, median total_score raw &
                                              relaxed, % total_score improved
  results/figures/relax_effect_overall.png       fa_rep + total_score raw-vs-
                                                  relaxed scatter, whole dataset
  results/figures/relax_effect_by_ligand.png     same two metrics, boxplots by ligand
  results/figures/relax_effect_by_backend.png    same two metrics, boxplots by backend
  results/figures/relax_effect_by_pose_cluster.png   fa_rep % reduction, one small
                                                      boxplot panel per protein, x =
                                                      that protein's own ca/pose clusters

Usage:
    python relax_effect_report.py
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")

# Mirrors scripts/cluster_conformations.py's BACKEND_PATTERNS -- duplicated
# here (not imported, same convention make_manifest.py already uses) just
# to recover which backend produced each complex_id, for the breakdown.
BACKEND_PATTERNS = {
    "alphafold3":   "alphafold3",
    "boltz":        "boltz",
    "chai1":        "chai",
    "openfold3":    "openfold",
    "protenix":     "protenix",
    "rosettafold3": "rosettafold",
}


def _backend_of(complex_id: str) -> str:
    for backend, pattern in BACKEND_PATTERNS.items():
        if pattern in complex_id:
            return backend
    return "unknown"


def load_per_complex_scores() -> pd.DataFrame:
    """One row per (complex_id, replica) -- fa_rep_raw/relaxed and
    total_raw/relaxed are repeated across every contact row for a complex
    in all_contacts.csv, so this only needs a handful of columns +
    dedup, not the full (often hundreds-of-MB) file in memory at once."""
    all_contacts_path = config.RESULTS_DIR / "all_contacts.csv"
    if not all_contacts_path.exists():
        sys.exit(f"{all_contacts_path} not found -- run aggregate.py first.")
    cols = ["complex_id", "protein", "ligand", "replica",
            "fa_rep_raw", "fa_rep_relaxed", "total_raw", "total_relaxed"]
    df = pd.read_csv(all_contacts_path, usecols=cols)
    df = df.drop_duplicates(["complex_id", "replica"]).reset_index(drop=True)
    df["backend"] = df["complex_id"].map(_backend_of)
    df["fa_rep_delta"] = df["fa_rep_relaxed"] - df["fa_rep_raw"]
    df["total_delta"] = df["total_relaxed"] - df["total_raw"]
    df["fa_rep_improved"] = df["fa_rep_delta"] < 0
    df["total_improved"] = df["total_delta"] < 0
    df["fa_rep_pct_reduction"] = 100 * (1 - df["fa_rep_relaxed"] / df["fa_rep_raw"])
    return df


def add_pose_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Join data/manifest.csv's ca_cluster/ligand_pose_cluster onto df by
    complex_id, and build a per-protein pose label ("ca<k>_pose<p>") --
    see module docstring for why this has to stay scoped per protein
    rather than pooled globally. Rows whose complex_id isn't in the
    manifest (e.g. all_contacts.csv is stale relative to a just-regenerated
    manifest.csv) or has no pose-cluster assignment (macro-state cluster
    too small for ligand-pose sub-clustering) are dropped -- print how many."""
    if not config.MANIFEST_CSV.exists():
        print(f"[relax_effect_report] {config.MANIFEST_CSV} not found -- skipping pose-cluster breakdown")
        return df.iloc[0:0]
    manifest = pd.read_csv(config.MANIFEST_CSV, usecols=["complex_id", "ca_cluster", "ligand_pose_cluster"])
    merged = df.merge(manifest, on="complex_id", how="inner")
    n_dropped = len(df) - len(merged)
    merged = merged.dropna(subset=["ca_cluster", "ligand_pose_cluster"])
    if n_dropped or len(merged) < len(df):
        print(f"[relax_effect_report] pose-cluster breakdown: {len(df) - len(merged)}/{len(df)} "
              "complex(es) dropped (not in manifest.csv, or no ligand-pose cluster assignment)")
    merged["pose_label"] = (
        "ca" + merged["ca_cluster"].astype(int).astype(str)
        + "_pose" + merged["ligand_pose_cluster"].astype(int).astype(str)
    )
    return merged


def write_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["ligand", "backend"])
        .agg(
            n=("complex_id", "count"),
            fa_rep_raw_median=("fa_rep_raw", "median"),
            fa_rep_relaxed_median=("fa_rep_relaxed", "median"),
            fa_rep_pct_improved=("fa_rep_improved", "mean"),
            total_raw_median=("total_raw", "median"),
            total_relaxed_median=("total_relaxed", "median"),
            total_pct_improved=("total_improved", "mean"),
        )
        .reset_index()
        .sort_values(["ligand", "backend"])
    )
    out = config.RESULTS_DIR / "relax_effect_summary.csv"
    summary.to_csv(out, index=False)
    print(f"[relax_effect_report] wrote {out} ({len(summary)} rows)")
    return summary


def plot_overall(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    ax.scatter(df["fa_rep_raw"], df["fa_rep_relaxed"], s=4, alpha=0.15, color="#1565C0", rasterized=True)
    lims = [min(df["fa_rep_raw"].min(), df["fa_rep_relaxed"].min()), df["fa_rep_raw"].quantile(0.999)]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="no change (y=x)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fa_rep, raw pose (REU)")
    ax.set_ylabel("fa_rep, after light relax (REU)")
    ax.set_title(f"fa_rep: improves in {df['fa_rep_improved'].mean():.1%} of complexes\n"
                 f"(median {df['fa_rep_raw'].median():.0f} -> {df['fa_rep_relaxed'].median():.0f} REU)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.scatter(df["total_raw"], df["total_relaxed"], s=4, alpha=0.15, color="#d62728", rasterized=True)
    lo = min(df["total_raw"].quantile(0.001), df["total_relaxed"].quantile(0.001))
    hi = max(df["total_raw"].quantile(0.995), df["total_relaxed"].quantile(0.995))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="no change (y=x)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("total_score, raw pose (REU)")
    ax.set_ylabel("total_score, after light relax (REU)")
    ax.set_title(f"total_score: WORSE in {(~df['total_improved']).mean():.1%} of complexes\n"
                 f"(median {df['total_raw'].median():.0f} -> {df['total_relaxed'].median():.0f} REU)")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Does the light, ligand-neighborhood-restricted FastRelax help? (n={len(df):,} complexes)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "relax_effect_overall.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[relax_effect_report] wrote {out}")


def _breakdown_figure(df: pd.DataFrame, group_col: str, out_name: str, title_suffix: str) -> None:
    order = df.groupby(group_col)["complex_id"].count().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.3 * len(order) + 3), 5.5))

    ax = axes[0]
    sns.boxplot(data=df, x=group_col, y="fa_rep_pct_reduction", order=order, ax=ax, showfliers=False, color="#1565C0")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(group_col)
    ax.set_ylabel("fa_rep % reduction after relax\n(higher = clash relief worked better)")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    lo, hi = df["total_delta"].quantile([0.01, 0.99])
    sns.boxplot(data=df, x=group_col, y="total_delta", order=order, ax=ax, showfliers=False, color="#d62728")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(min(lo, 0), hi)
    ax.set_xlabel(group_col)
    ax.set_ylabel("total_score delta after relax (REU)\n(negative = improved; outliers beyond the\n1st/99th percentile clipped for readability)")
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Clash-relief effect by {title_suffix} (box = IQR, whiskers = 1.5x IQR, outliers hidden)")
    fig.tight_layout()
    out = config.FIGURES_DIR / out_name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[relax_effect_report] wrote {out}")


def plot_by_pose_cluster(df: pd.DataFrame) -> None:
    """One small boxplot panel per protein, x = that protein's own
    "ca<k>_pose<p>" labels (see add_pose_cluster_labels) -- pose-cluster
    ids aren't comparable across proteins, so this is deliberately faceted
    rather than pooled the way the ligand/backend breakdowns are."""
    pdf = add_pose_cluster_labels(df)
    if pdf.empty:
        print("[relax_effect_report] no pose-cluster data available, skipping that breakdown")
        return

    proteins = sorted(pdf["protein"].unique())
    n_cols = 6
    n_rows = -(-len(proteins) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * n_rows), squeeze=False)

    for i, protein in enumerate(proteins):
        ax = axes[i // n_cols][i % n_cols]
        sub = pdf[pdf["protein"] == protein]
        order = sorted(sub["pose_label"].unique())
        sns.boxplot(data=sub, x="pose_label", y="fa_rep_pct_reduction", order=order,
                    ax=ax, showfliers=False, color="#1565C0", width=0.6)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(protein, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(len(proteins), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.supylabel("fa_rep % reduction after relax (higher = clash relief worked better)", fontsize=10)
    fig.suptitle("Clash-relief effect by ligand-pose cluster, per protein "
                  "(box = IQR, whiskers = 1.5x IQR, outliers hidden)", fontsize=12)
    fig.tight_layout(rect=(0.01, 0, 1, 0.98))
    out = config.FIGURES_DIR / "relax_effect_by_pose_cluster.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[relax_effect_report] wrote {out} ({len(proteins)} protein panel(s))")


def main():
    df = load_per_complex_scores()
    print(f"[relax_effect_report] loaded {len(df)} complex(es), "
          f"{df['ligand'].nunique()} ligand(s), {df['backend'].nunique()} backend(s)")
    write_summary(df)
    plot_overall(df)
    _breakdown_figure(df, "ligand", "relax_effect_by_ligand.png", "ligand")
    _breakdown_figure(df, "backend", "relax_effect_by_backend.png", "backend")
    plot_by_pose_cluster(df)


if __name__ == "__main__":
    main()
