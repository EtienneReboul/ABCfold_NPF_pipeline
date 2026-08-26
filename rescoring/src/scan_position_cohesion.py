#!/usr/bin/env python3
"""
rescoring/src/scan_position_cohesion.py
===========================================
Systematic version of the ad hoc position-by-position check done by hand
in conversation (positions 2, 20, 27, prompted by eyeballing
`haddock_redocking_exploration.ipynb`'s LDA-vs-Rosetta-energy scatter): for
EVERY CDD pocket position (1-35), how cleanly does Rosetta's own redocked-
pose energetics separate GA1 importers from non-importers, and does that
line up with NPF_LDA_kernel's own sequence-only importance for that
position?

Reads `redocking/results/rescoring/lda_unfavorable_contacts.csv`
(rescore_redocked_aggregate.py's own output: one row per (protein, role,
position) actually contacted in >=1 redocked complex, with that protein's
mean ligand<->residue twobody_total at that position) -- no
re-aggregation, this scan just re-slices data that already exists.

For each position (with enough proteins on both sides to mean anything --
see --min-importer-n/--min-non-importer-n):
  - frac_unfavorable per role (fraction of that role's proteins with
    mean_twobody_total > 0 there)
  - mean twobody_total per role, and the gap between them
  - Mann-Whitney U p-value (two-sided) on mean_twobody_total, importer vs
    non_importer -- despite small importer n (<=5 here), a position with a
    genuinely clean split (e.g. position 20's 0/5 vs 12/19 unfavorable)
    still produces a small p-value. Treat this as a RANKING heuristic, not
    a multiple-testing-corrected significance claim -- n=5 per position on
    the importer side has real limits, see the module's own printed
    caveat.
  - direction: "importer_favorable" (the biologically sensible direction --
    importer more stable/negative than non_importer there) vs
    "non_importer_favorable" (the reverse -- worth flagging as
    counter-intuitive, not silently discarding, see position 27 in
    conversation, which turned out unfavorable for BOTH roles almost
    equally and is excluded by the recommendation filter below for that
    reason, not because the direction was wrong).
  - the dominant NPF_LDA_kernel Z-scale driver at that position (from
    `data/lda_GA1_loadings.tsv`) -- e.g. position 20's signal is driven by
    steric bulk (Z2), which maps directly onto Rosetta's fa_rep/fa_atr;
    position 2's is driven by proline/turn-propensity (Z5), a backbone
    descriptor a static-pose two-body sidechain energy term isn't expected
    to capture well -- this is why some LDA-important positions translate
    into a clean Rosetta-energetic split and others don't.

Output: `redocking/results/rescoring/position_cohesion_scan.csv` (every
position that clears the coverage minimums, sorted by Mann-Whitney p),
plus a printed "recommended for experimental follow-up" shortlist:
p <= --p-threshold, direction == importer_favorable, gap >=
--min-gap, and importer side mostly favorable (frac_unfavorable_importer
<= --max-importer-unfavorable-frac).
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import mannwhitneyu

import config

LDA_UNFAVORABLE_CSV = config.PIPELINE_ROOT / "redocking" / "results" / "rescoring" / "lda_unfavorable_contacts.csv"
OUT_CSV = config.PIPELINE_ROOT / "redocking" / "results" / "rescoring" / "position_cohesion_scan.csv"
LDA_LOADINGS_TSV = config.DATA_DIR / "lda_GA1_loadings.tsv"


def dominant_z_scale(loadings: pd.DataFrame, position: float) -> tuple[str, float]:
    sub = loadings[loadings["position"] == position]
    if sub.empty:
        return "?", float("nan")
    row = sub.loc[sub["lda_coef"].abs().idxmax()]
    return row["z_name"], float(row["lda_coef"])


def scan(lda_unfavorable: pd.DataFrame, loadings: pd.DataFrame,
         min_importer_n: int, min_non_importer_n: int) -> pd.DataFrame:
    rows = []
    for position, group in lda_unfavorable.groupby("position"):
        imp = group[group["role"] == "importer"]
        non = group[group["role"] == "non_importer"]
        if len(imp) < min_importer_n or len(non) < min_non_importer_n:
            continue

        frac_unfav_imp = imp["unfavorable"].mean()
        frac_unfav_non = non["unfavorable"].mean()
        mean_imp = imp["mean_twobody_total"].mean()
        mean_non = non["mean_twobody_total"].mean()

        try:
            _, p = mannwhitneyu(imp["mean_twobody_total"], non["mean_twobody_total"], alternative="two-sided")
        except ValueError:
            p = float("nan")

        z_name, z_coef = dominant_z_scale(loadings, position)
        rows.append(dict(
            position=position,
            n_importer=len(imp), n_non_importer=len(non),
            frac_unfavorable_importer=frac_unfav_imp, frac_unfavorable_non_importer=frac_unfav_non,
            unfavorable_gap=frac_unfav_non - frac_unfav_imp,
            mean_twobody_importer=mean_imp, mean_twobody_non_importer=mean_non,
            delta_mean=mean_non - mean_imp,
            mannwhitney_p=p,
            direction="importer_favorable" if mean_imp < mean_non else "non_importer_favorable",
            lda_importance=group["lda_importance"].iloc[0],
            dominant_z_scale=z_name, dominant_z_coef=z_coef,
        ))
    return pd.DataFrame(rows).sort_values("mannwhitney_p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-importer-n", type=int, default=4,
                     help="minimum importer proteins with a contact at a position to consider it (default 4/5)")
    ap.add_argument("--min-non-importer-n", type=int, default=12,
                     help="minimum non_importer proteins with a contact at a position to consider it (default 12/19)")
    ap.add_argument("--p-threshold", type=float, default=0.1,
                     help="Mann-Whitney p cutoff for the recommendation shortlist (default 0.1 -- "
                          "exploratory ranking, not a corrected significance threshold, see module docstring)")
    ap.add_argument("--min-gap", type=float, default=0.3,
                     help="minimum (frac_unfavorable_non_importer - frac_unfavorable_importer) for the shortlist")
    ap.add_argument("--max-importer-unfavorable-frac", type=float, default=0.25,
                     help="importer side must be mostly favorable (default: at most 25% of importers unfavorable)")
    args = ap.parse_args()

    lda_unfavorable = pd.read_csv(LDA_UNFAVORABLE_CSV)
    loadings = pd.read_csv(LDA_LOADINGS_TSV, sep="\t")

    result = scan(lda_unfavorable, loadings, args.min_importer_n, args.min_non_importer_n)
    result.to_csv(OUT_CSV, index=False)
    print(f"[scan_position_cohesion] wrote {OUT_CSV} ({len(result)}/35 positions cleared the coverage minimums)")
    print("[scan_position_cohesion] CAVEAT: importer n<=5 per position -- Mann-Whitney p-values here are a "
          "ranking heuristic across positions, not multiple-testing-corrected significance claims. Treat this "
          "as a shortlist to prioritize experimental follow-up, not a finished result.\n")

    shortlist = result[
        (result["mannwhitney_p"] <= args.p_threshold)
        & (result["direction"] == "importer_favorable")
        & (result["unfavorable_gap"] >= args.min_gap)
        & (result["frac_unfavorable_importer"] <= args.max_importer_unfavorable_frac)
    ]
    print(f"[scan_position_cohesion] {len(shortlist)} position(s) recommended for experimental follow-up "
          f"(p<={args.p_threshold}, importer-favorable direction, unfavorable_gap>={args.min_gap}, "
          f"importer unfavorable frac<={args.max_importer_unfavorable_frac}):\n")
    cols = ["position", "n_importer", "n_non_importer", "frac_unfavorable_importer",
            "frac_unfavorable_non_importer", "mannwhitney_p", "lda_importance", "dominant_z_scale"]
    print(shortlist[cols].to_string(index=False))

    print("\n[scan_position_cohesion] full ranking (top 15 by p-value):")
    print(result[cols + ["direction"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
