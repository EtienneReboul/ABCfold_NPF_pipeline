"""
mmgbsa/src/plots.py
===================
Stage 8: static SVG figures from the Stage 5-7 tables. Same headless
matplotlib/seaborn stack as rescoring/src/plots.py -- no notebook, no
widgets, one .svg per figure in results/mmgbsa/figures/.

Figures:
  gb_profile_pocket.svg        per-CDD-position mean GB total, importer vs
                               non_importer, with inter-protein SEM error bars
  gb_vs_rosetta_scatter.svg    per-(protein,position) GB total vs Rosetta
                               two-body total, colored by role
  dg_gb_by_role.svg            per-complex dG_GB distribution by role
  gb_top_residues.svg          top |GB total| CDD positions, importer bars with
                               inter-replica SD whiskers
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    out = config.FIGURES_DIR / name
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[stage8] wrote {out}")


def gb_profile_pocket(decomp_pos: pd.DataFrame) -> None:
    d = decomp_pos[decomp_pos["is_cdd_pocket"].astype(str).str.lower().isin(("true", "1"))].copy()
    if d.empty:
        print("[stage8] no CDD-pocket rows -- skipping gb_profile_pocket")
        return
    g = (d.groupby(["cdd_position", "role"])["mean_gb_total"]
         .agg(["mean", "sem"]).reset_index())
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * g["cdd_position"].nunique()), 4.5))
    for role, sub in g.groupby("role"):
        ax.errorbar(sub["cdd_position"], sub["mean"], yerr=sub["sem"], marker="o",
                    capsize=3, label=role, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("CDD pocket position")
    ax.set_ylabel("mean GB contribution (kcal/mol)")
    ax.set_title("Per-residue GB decomposition across the CDD pocket")
    ax.legend()
    _save(fig, "gb_profile_pocket.svg")


def gb_vs_rosetta_scatter() -> None:
    comp = config.MMGBSA_DIR / "engine_comparison.csv"
    if not comp.exists():
        print("[stage8] engine_comparison.csv missing -- run compare_engines.py; skipping scatter")
        return
    df = pd.read_csv(comp).dropna(subset=["mean_gb_total", "mean_twobody_total"])
    if df.empty:
        print("[stage8] no overlapping GB/Rosetta positions -- skipping scatter")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sns.scatterplot(data=df, x="mean_twobody_total", y="mean_gb_total", hue="role", s=18, ax=ax)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Rosetta two-body total (REU)")
    ax.set_ylabel("GB total (kcal/mol)")
    ax.set_title("Per-position: MM-GBSA vs PyRosetta")
    _save(fig, "gb_vs_rosetta_scatter.svg")


def dg_gb_by_role() -> None:
    if not config.BINDING_ENERGY_SUMMARY_CSV.exists():
        print("[stage8] binding_energy_summary.csv missing -- skipping dg_gb_by_role")
        return
    be = pd.read_csv(config.BINDING_ENERGY_SUMMARY_CSV)
    perc = be.groupby(["complex_id", "role"])["dg_gb_total"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.boxplot(data=perc, x="role", y="dg_gb_total", ax=ax)
    sns.stripplot(data=perc, x="role", y="dg_gb_total", color="black", size=3, alpha=0.5, ax=ax)
    ax.set_ylabel("dG_GB (kcal/mol, mean over replicas)")
    ax.set_title("MM-GBSA binding free energy by role")
    _save(fig, "dg_gb_by_role.svg")


def gb_top_residues(decomp_pos: pd.DataFrame, n: int = 15) -> None:
    d = decomp_pos[(decomp_pos["is_cdd_pocket"].astype(str).str.lower().isin(("true", "1")))
                   & (decomp_pos["role"] == "importer")].copy()
    if d.empty:
        print("[stage8] no importer CDD rows -- skipping gb_top_residues")
        return
    g = d.groupby("cdd_position").agg(mean_gb_total=("mean_gb_total", "mean"),
                                      sem=("sem", "mean")).reset_index()
    g["abs"] = g["mean_gb_total"].abs()
    g = g.sort_values("abs", ascending=False).head(n).sort_values("mean_gb_total")
    fig, ax = plt.subplots(figsize=(5.5, max(3, 0.35 * len(g))))
    ax.barh(g["cdd_position"].astype(str), g["mean_gb_total"], xerr=g["sem"], capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mean GB contribution (kcal/mol)")
    ax.set_ylabel("CDD pocket position")
    ax.set_title(f"Top {len(g)} GB-contributing CDD positions (importers)")
    _save(fig, "gb_top_residues.svg")


def main() -> None:
    if not config.DECOMP_BY_POSITION_CSV.exists():
        raise SystemExit(f"{config.DECOMP_BY_POSITION_CSV} not found -- run aggregate_decomp.py first.")
    decomp_pos = pd.read_csv(config.DECOMP_BY_POSITION_CSV)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gb_profile_pocket(decomp_pos)
    gb_vs_rosetta_scatter()
    dg_gb_by_role()
    gb_top_residues(decomp_pos)


if __name__ == "__main__":
    main()
