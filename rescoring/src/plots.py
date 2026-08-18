#!/usr/bin/env python3
"""
rescoring/src/plots.py
=========================
Plots from aggregate.py's outputs:
  (a) per-residue stacked bar of score terms for one representative complex
      (overall, and one per protein)
  (b) residue x protein heatmap of mean two-body totals, one per ligand
      category (positions are only comparable within one ligand's own
      Rosetta energetics, not across chemically different ligands)
  (c) Rosetta-hotspot-vs-LDA-importance scatter, one per ligand category
      that has a fit_pocket_lda.py output — the direct visual answer to
      "do the positions Rosetta finds energetically important line up with
      the ones the sequence-only LDA flagged?"

Generalized from the sibling project's version: replaces its
importer-vs-non-importer class-difference bar chart (not reproducible here
— see aggregate.py's module docstring) with the Rosetta-vs-LDA scatter.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")


def _complex_totals(contacts: pd.DataFrame) -> pd.Series:
    """complex_id -> summed twobody_total, one edge per (complex_id, replica, position)."""
    return (
        contacts.dropna(subset=["position"])
        .drop_duplicates(["complex_id", "replica", "position"])
        .groupby("complex_id")["twobody_total"].sum()
    )


def _draw_stacked_bar(sub: pd.DataFrame, title: str, out: Path) -> None:
    pivot = sub.pivot_table(index="position", columns="scoretype", values="weighted_energy", aggfunc="mean").fillna(0)
    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(pivot)), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("pocket position")
    ax.set_ylabel("weighted energy (REU)")
    ax.set_title(title)
    ax.legend(title="scoretype", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_representative_stacked_bar(contacts: pd.DataFrame, ligand: str | None = None,
                                     complex_id: str | None = None) -> None:
    scoped = contacts if ligand is None else contacts[contacts["ligand"] == ligand]
    if complex_id is None:
        complex_id = _complex_totals(scoped).idxmax()

    sub = contacts[contacts["complex_id"] == complex_id].dropna(subset=["position"])
    name = "representative_complex_stacked_bar" if ligand is None else f"representative_complex_stacked_bar_{ligand}"
    out = config.FIGURES_DIR / f"{name}.png"
    _draw_stacked_bar(sub, f"Score-term decomposition -- {complex_id}", out)
    print(f"[plots] wrote {out} (complex_id={complex_id})")


def plot_representative_stacked_bar_per_protein(contacts: pd.DataFrame) -> None:
    """One stacked-bar plot per protein (its own highest-total-unfavorable-
    contribution complex), filed under
    results/figures/representative_stacked_bars/<ligand>/<protein>.png."""
    mapped = contacts.dropna(subset=["position"])
    out_dir = config.FIGURES_DIR / "representative_stacked_bars"

    n_written = 0
    for protein, group in mapped.groupby("protein"):
        ligand = group["ligand"].iloc[0]
        complex_id = _complex_totals(group).idxmax()
        sub = group[group["complex_id"] == complex_id]
        out = out_dir / ligand / f"{protein}.png"
        _draw_stacked_bar(sub, f"Score-term decomposition -- {protein} ({ligand})\n{complex_id}", out)
        n_written += 1

    print(f"[plots] wrote {n_written} per-protein stacked bars under {out_dir}/<ligand>/")


def plot_residue_by_protein_heatmap(contacts: pd.DataFrame, ligand: str) -> None:
    mapped = (
        contacts[contacts["ligand"] == ligand]
        .dropna(subset=["position"])
        .drop_duplicates(["complex_id", "replica", "position"])
    )
    if mapped.empty:
        print(f"[plots] {ligand}: no mapped-position contacts, skipping heatmap")
        return
    per_protein = mapped.groupby(["protein", "position"])["twobody_total"].mean().reset_index()
    pivot = per_protein.pivot(index="protein", columns="position", values="twobody_total")

    fig, ax = plt.subplots(figsize=(max(10, 0.3 * pivot.shape[1]), max(4, 0.3 * pivot.shape[0])))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "mean two-body total (REU)"})
    ax.set_xlabel("pocket position")
    ax.set_ylabel("protein")
    ax.set_title(f"Ligand<->residue two-body total, mean per protein -- {ligand}")
    fig.tight_layout()
    out = config.FIGURES_DIR / f"residue_by_protein_heatmap_{ligand}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] wrote {out}")


def plot_rosetta_vs_lda_scatter(overlay: pd.DataFrame, ligand: str) -> None:
    """One point per pocket position: x = sequence-LDA importance
    (fit_pocket_lda.py), y = this ligand's mean Rosetta twobody_total
    (aggregate.py's residue_rank) -- the direct visual answer to "do
    Rosetta's own energetic hotspots line up with the positions the
    sequence-only classifier flagged?" Skipped for ligands with no
    fit_pocket_lda.py output (all-NaN lda_importance -- see
    fit_pocket_lda.py's MIN_POSITIVES)."""
    sub = overlay[overlay["ligand"] == ligand].dropna(subset=["lda_importance"])
    if sub.empty:
        print(f"[plots] {ligand}: no lda_importance (no fit_pocket_lda.py output for this "
              "ligand) -- skipping Rosetta-vs-LDA scatter")
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in sub["mean"]]
    ax.scatter(sub["lda_importance"], sub["mean"], c=colors, s=60, edgecolor="black", linewidth=0.5)
    for _, row in sub.iterrows():
        ax.annotate(str(int(row["position"])), (row["lda_importance"], row["mean"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("sequence-LDA importance (|coef| summed over 5 Z-scales)")
    ax.set_ylabel("mean Rosetta ligand<->residue two-body total (REU)")
    ax.set_title(f"Rosetta hotspots vs. sequence-LDA importance -- {ligand}\n"
                 "(green = Rosetta-stabilizing, red = Rosetta-unfavorable; labels = pocket position)",
                 fontsize=10)
    fig.tight_layout()
    out = config.FIGURES_DIR / f"rosetta_vs_lda_scatter_{ligand}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] wrote {out}")


def main():
    all_contacts_path = config.RESULTS_DIR / "all_contacts.csv"
    overlay_path = config.RESULTS_DIR / "lda_overlay.csv"
    if not all_contacts_path.exists() or not overlay_path.exists():
        sys.exit("Missing aggregate.py outputs -- run aggregate.py first.")

    contacts = pd.read_csv(all_contacts_path)
    overlay = pd.read_csv(overlay_path)
    ligands = sorted(contacts["ligand"].unique())

    plot_representative_stacked_bar(contacts)
    plot_representative_stacked_bar_per_protein(contacts)
    for ligand in ligands:
        plot_residue_by_protein_heatmap(contacts, ligand)
        plot_rosetta_vs_lda_scatter(overlay, ligand)


if __name__ == "__main__":
    main()
