#!/usr/bin/env python3
"""
rescoring/src/plip_analysis.py
==================================
Does PLIP's own explicit protein-ligand interaction detection (hydrogen
bonds, salt bridges, hydrophobic contacts, pi-stacking, water bridges, ...)
on the ChimeraX-minimized poses agree with the CDD/InterPro-defined 35-
position putative binding site (data/position_resnr_map.csv, see
build_position_mapping.py) that the sequence-only LDA classifier
(fit_pocket_lda.py) also uses? A third, independent line of evidence
alongside the Rosetta energetic hotspots (aggregate.py/plots.py) already
in this pipeline -- PLIP flags a residue only when it makes an actual,
geometrically-defined interaction (not just "within some cutoff", the way
decompose.py's Rosetta two-body-energy contacts are scoped), so it's a
different kind of check on the same question.

Same residue-numbering space throughout: pose_prep.py never renumbers the
protein chain (it's the ABCfold CIF's own numbering, untouched through
staging and ChimeraX minimization), and decompose.py's own `prot_resi`
column is `pose.pdb_info().number(r)` -- i.e. also the original PDB
numbering, not Rosetta's internal pose numbering. So PLIP's own `resnr`
field (read straight off the minimized PDB) is directly comparable to
position_resnr_map.csv's `resnr` with no renumbering step needed.

Two outputs:
  results/plip_cdd_agreement.csv   per protein: of the residues PLIP
                                    actually flags as making a real
                                    interaction with the ligand (pooled
                                    across that protein's complexes), how
                                    many fall inside vs. outside the 35 CDD
                                    pocket positions (precision), and of
                                    the 35 CDD positions, how many are ever
                                    actually contacted (recall)
  results/plip_lda_overlay.csv     per (ligand, position): PLIP contact
                                    frequency across that ligand's
                                    complexes, next to sequence-LDA
                                    importance (where fit_pocket_lda.py has
                                    one) -- same "positions the sequence
                                    classifier flags vs. positions physical
                                    contact detection flags" comparison
                                    aggregate.py/plots.py already runs for
                                    Rosetta energetics.

Run after plip_run_batch.py has produced results/plip/*_report.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pliparser.plip2csv import plip2dictlist

import config

sns.set_theme(style="whitegrid")

PLIP_DIR = config.RESULTS_DIR / "plip"


def load_plip_contacts() -> pd.DataFrame:
    """One row per (complex_id, receptor resnr) actually involved in ANY
    PLIP interaction type -- deduped across interaction types/rows."""
    manifest = pd.read_csv(config.MANIFEST_CSV)[["complex_id", "protein", "ligand"]]

    rows = []
    for report_path in sorted(PLIP_DIR.glob("*_report.txt")):
        complex_id = report_path.name[:-len("_report.txt")]
        plip_dict = plip2dictlist(report_path)
        residues: dict[int, dict] = {}
        for interaction_type, entries in plip_dict.items():
            for row in entries:
                reschain = row.get("reschain", config.PROTEIN_CHAIN)
                if reschain != config.PROTEIN_CHAIN:
                    continue
                try:
                    resnr = int(row["resnr"])
                except (KeyError, ValueError):
                    continue
                d = residues.setdefault(resnr, {"restype": row.get("restype", ""), "types": set()})
                d["types"].add(interaction_type)
        for resnr, d in residues.items():
            rows.append({
                "complex_id": complex_id,
                "resnr": resnr,
                "restype": d["restype"],
                "interaction_types": ";".join(sorted(d["types"])),
            })

    if not rows:
        sys.exit(f"No PLIP reports found in {PLIP_DIR} -- run plip_run_batch.py first.")

    contacts = pd.DataFrame(rows).merge(manifest, on="complex_id", how="left")
    n_unmatched = contacts["protein"].isna().sum()
    if n_unmatched:
        print(f"[plip_analysis] NOTE: {n_unmatched} contact row(s) have no manifest.csv match "
              "(stale report from a since-changed manifest?) -- dropped")
        contacts = contacts.dropna(subset=["protein"])
    return contacts


def add_position(contacts: pd.DataFrame) -> pd.DataFrame:
    if not config.POSITION_RESNR_MAP_CSV.exists():
        sys.exit(f"{config.POSITION_RESNR_MAP_CSV} not found -- run build_position_mapping.py first.")
    position_map = pd.read_csv(config.POSITION_RESNR_MAP_CSV).dropna(subset=["resnr"])
    position_map["resnr"] = position_map["resnr"].astype(int)
    merged = contacts.merge(position_map, on=["protein", "resnr"], how="left")
    n_unmapped = merged["position"].isna().sum()
    if n_unmapped:
        print(f"[plip_analysis] NOTE: {n_unmapped}/{len(merged)} PLIP contact rows are outside the "
              "35 CDD pocket positions (real interactions PLIP found that the CDD/LDA definition "
              "didn't flag) -- kept, with position=NaN, for the CDD-agreement precision/recall below.")
    return merged


def cdd_agreement(contacts_with_position: pd.DataFrame) -> pd.DataFrame:
    """Per protein: does PLIP's own explicit-interaction residue set line up
    with the CDD's 35 putative pocket positions, or disagree?"""
    position_map = pd.read_csv(config.POSITION_RESNR_MAP_CSV).dropna(subset=["resnr"])

    rows = []
    for protein, group in contacts_with_position.groupby("protein"):
        unique_residues = group.drop_duplicates(["resnr"])
        n_contacted = len(unique_residues)
        n_in_cdd = int(unique_residues["position"].notna().sum())
        n_out_cdd = n_contacted - n_in_cdd
        precision = n_in_cdd / n_contacted if n_contacted else float("nan")

        protein_positions = position_map[position_map["protein"] == protein]
        n_cdd_total = protein_positions["position"].nunique()
        contacted_positions = set(unique_residues["position"].dropna().unique())
        n_cdd_contacted = len(contacted_positions & set(protein_positions["position"]))
        recall = n_cdd_contacted / n_cdd_total if n_cdd_total else float("nan")

        rows.append(dict(
            protein=protein, ligand=group["ligand"].iloc[0],
            n_complexes=group["complex_id"].nunique(),
            n_unique_contacted_residues=n_contacted,
            n_in_cdd_pocket=n_in_cdd, n_outside_cdd_pocket=n_out_cdd, precision=precision,
            n_cdd_positions_total=n_cdd_total, n_cdd_positions_contacted=n_cdd_contacted, recall=recall,
        ))
    return pd.DataFrame(rows).sort_values(["ligand", "protein"])


def _load_importance(ligand: str) -> pd.DataFrame | None:
    path = config.DATA_DIR / f"position_importance_{ligand}.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def plip_lda_overlay(contacts_with_position: pd.DataFrame) -> pd.DataFrame:
    """Per (ligand, position): fraction of that ligand's PLIP-analyzed
    complexes in which the position was ever contacted, overlaid with
    sequence-LDA importance (mirrors aggregate.py's lda_overlay for
    Rosetta energetics)."""
    total_per_ligand = contacts_with_position.groupby("ligand")["complex_id"].nunique()

    mapped = contacts_with_position.dropna(subset=["position"])
    per_complex_position = mapped.drop_duplicates(["complex_id", "position"])
    freq = (
        per_complex_position.groupby(["ligand", "position"])["complex_id"].nunique()
        .reset_index(name="n_complexes_with_contact")
    )
    freq["n_complexes_total"] = freq["ligand"].map(total_per_ligand)
    freq["contact_frequency"] = freq["n_complexes_with_contact"] / freq["n_complexes_total"]

    frames = []
    for ligand, group in freq.groupby("ligand"):
        importance = _load_importance(ligand)
        if importance is None:
            print(f"[plip_analysis] {ligand}: no fit_pocket_lda.py output -- overlay is "
                  "PLIP-contact-only (no importance column) for this ligand")
            merged = group.copy()
            merged["lda_importance"] = float("nan")
        else:
            merged = group.merge(importance, on="position", how="left")
            merged = merged.rename(columns={"importance": "lda_importance"})
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["ligand", "position"])


def plot_cdd_agreement_bar(agreement: pd.DataFrame) -> None:
    sub = agreement.sort_values(["ligand", "protein"])
    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(sub)), 5))
    x = range(len(sub))
    ax.bar([i - 0.2 for i in x], sub["precision"], width=0.4, label="precision "
           "(contacted residues that ARE in the CDD pocket)", color="#1f77b4")
    ax.bar([i + 0.2 for i in x], sub["recall"], width=0.4, label="recall "
           "(CDD pocket positions PLIP ever actually contacts)", color="#ff7f0e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["protein"], rotation=90, fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("PLIP explicit-interaction residues vs. the CDD-defined 35-position putative binding site")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=1, fontsize=8)
    fig.tight_layout()
    out = config.FIGURES_DIR / "plip_cdd_agreement.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plip_analysis] wrote {out}")


def plot_plip_vs_lda_scatter(overlay: pd.DataFrame, ligand: str) -> None:
    sub = overlay[overlay["ligand"] == ligand].dropna(subset=["lda_importance"])
    if sub.empty:
        print(f"[plip_analysis] {ligand}: no lda_importance -- skipping PLIP-vs-LDA scatter")
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(sub["lda_importance"], sub["contact_frequency"], s=60,
               c="#1f77b4", edgecolor="black", linewidth=0.5)
    for _, row in sub.iterrows():
        ax.annotate(str(int(row["position"])), (row["lda_importance"], row["contact_frequency"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("sequence-LDA importance (|coef| summed over 5 Z-scales)")
    ax.set_ylabel("PLIP contact frequency (fraction of complexes with a real interaction at this position)")
    ax.set_title(f"PLIP contact frequency vs. sequence-LDA importance -- {ligand}\n"
                 "(labels = pocket position)", fontsize=10)
    fig.tight_layout()
    out = config.FIGURES_DIR / f"plip_vs_lda_scatter_{ligand}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plip_analysis] wrote {out}")


def main():
    contacts = load_plip_contacts()
    print(f"[plip_analysis] loaded PLIP contacts for {contacts['complex_id'].nunique()} complexes, "
          f"{contacts['protein'].nunique()} proteins, ligand(s): {sorted(contacts['ligand'].unique())}")
    contacts = add_position(contacts)
    contacts.to_csv(config.RESULTS_DIR / "plip_contacts.csv", index=False)

    agreement = cdd_agreement(contacts)
    agreement.to_csv(config.RESULTS_DIR / "plip_cdd_agreement.csv", index=False)
    print(f"[plip_analysis] wrote plip_cdd_agreement.csv ({len(agreement)} proteins) -- "
          f"pooled precision {agreement['n_in_cdd_pocket'].sum() / agreement['n_unique_contacted_residues'].sum():.2f}, "
          f"pooled recall {agreement['n_cdd_positions_contacted'].sum() / agreement['n_cdd_positions_total'].sum():.2f}")
    plot_cdd_agreement_bar(agreement)

    overlay = plip_lda_overlay(contacts)
    overlay.to_csv(config.RESULTS_DIR / "plip_lda_overlay.csv", index=False)
    print(f"[plip_analysis] wrote plip_lda_overlay.csv ({len(overlay)} rows)")
    for ligand in sorted(overlay["ligand"].unique()):
        plot_plip_vs_lda_scatter(overlay, ligand)


if __name__ == "__main__":
    main()
