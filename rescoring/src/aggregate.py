#!/usr/bin/env python3
"""
rescoring/src/aggregate.py
=============================
Pool every per-complex tidy table, map each protein's own residue numbering
onto the common pocket "position" (1-35, data/position_resnr_map.csv — see
build_position_mapping.py), and produce:

  results/all_contacts.csv      pooled long table (every row from every
                                 results/per_complex/*.csv)
  results/residue_rank.csv      per (ligand, position): mean/std/n
                                 twobody_total, ranked -- this pipeline's
                                 own Rosetta "hotspot" ranking, one per
                                 ligand category (see module docstring)
  results/lda_overlay.csv       residue_rank left-joined with
                                 data/position_importance_<ligand>.tsv
                                 (fit_pocket_lda.py), where a fit exists for
                                 that ligand -- NaN importance columns for
                                 ligands with too few assigned proteins to
                                 fit (see fit_pocket_lda.py's MIN_POSITIVES)

Generalized from the sibling project's version, which compared Rosetta
energetics between GA1-docked importers and GA1-docked non-importers — not
reproducible here, since this pipeline only ever co-folds a protein with ITS
OWN assigned ligand (a nitrate transporter is never GA1-docked here), so
there's no second class sharing the same ligand to compare against. Instead:
for each ligand category, rank that category's own real Rosetta
ligand<->residue energetics by pocket position, and overlay them against
that category's own sequence-derived LDA importance (real published
classifier for GA1, freshly one-vs-rest-fit for nitrate/ABA — see
fit_pocket_lda.py) — "do the positions Rosetta finds energetically
important line up with the ones the sequence-only classifier flagged?"

Run after run_batch.py has produced at least some results/per_complex/*.csv,
and fit_pocket_lda.py has run (for the overlay step; residue_rank.csv is
still produced without it, just with no importance columns to join).
"""
import sys

import pandas as pd

import config


def load_all_contacts() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(config.PER_COMPLEX_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        sys.exit(f"No per-complex results found in {config.PER_COMPLEX_DIR} -- run run_batch.py first.")
    return pd.concat(frames, ignore_index=True)


def add_position(contacts: pd.DataFrame) -> pd.DataFrame:
    if not config.POSITION_RESNR_MAP_CSV.exists():
        sys.exit(f"{config.POSITION_RESNR_MAP_CSV} not found -- run build_position_mapping.py first.")
    position_map = pd.read_csv(config.POSITION_RESNR_MAP_CSV)
    merged = contacts.merge(
        position_map, left_on=["protein", "prot_resi"], right_on=["protein", "resnr"], how="left",
    )
    n_unmapped = merged["position"].isna().sum()
    if n_unmapped:
        print(f"[aggregate] NOTE: {n_unmapped}/{len(merged)} contact rows are outside the "
              "35 pocket positions (residues Rosetta found in contact with the ligand "
              "but that CDD/the LDA didn't flag) -- kept, with position=NaN, for the "
              "full picture in all_contacts.csv, but they drop out of the position-indexed "
              "aggregates below.")
    return merged


def residue_rank(contacts: pd.DataFrame) -> pd.DataFrame:
    mapped = contacts.dropna(subset=["position"])
    # one twobody_total per (complex_id, replica, position) -- dedupe across scoretype rows
    per_edge = mapped.drop_duplicates(["complex_id", "replica", "position"])
    rank = (
        per_edge.groupby(["ligand", "position"])["twobody_total"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    return rank.sort_values(["ligand", "mean"])


def _load_importance(ligand: str) -> pd.DataFrame | None:
    path = config.DATA_DIR / f"position_importance_{ligand}.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def lda_overlay(rank: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ligand, group in rank.groupby("ligand"):
        importance = _load_importance(ligand)
        if importance is None:
            print(f"[aggregate] {ligand}: no fit_pocket_lda.py output -- overlay is "
                  "Rosetta-hotspot-only (no importance column) for this ligand")
            merged = group.copy()
            merged["lda_importance"] = float("nan")
        else:
            merged = group.merge(importance, on="position", how="left")
            merged = merged.rename(columns={"importance": "lda_importance"})
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["ligand", "position"])


def main():
    contacts = load_all_contacts()
    print(f"[aggregate] loaded {len(contacts)} rows from "
          f"{contacts['complex_id'].nunique()} complexes, ligand(s): {sorted(contacts['ligand'].unique())}")
    contacts = add_position(contacts)
    contacts.to_csv(config.RESULTS_DIR / "all_contacts.csv", index=False)

    rank = residue_rank(contacts)
    rank.to_csv(config.RESULTS_DIR / "residue_rank.csv", index=False)
    print(f"[aggregate] wrote residue_rank.csv ({len(rank)} rows)")

    overlay = lda_overlay(rank)
    overlay.to_csv(config.RESULTS_DIR / "lda_overlay.csv", index=False)
    print(f"[aggregate] wrote lda_overlay.csv ({len(overlay)} rows)")


if __name__ == "__main__":
    main()
