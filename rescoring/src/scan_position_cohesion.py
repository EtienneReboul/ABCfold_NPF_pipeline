#!/usr/bin/env python3
"""
rescoring/src/scan_position_cohesion.py
===========================================
Systematic version of the ad hoc position-by-position check done by hand
in conversation (positions 2, 20, 27, prompted by eyeballing
`haddock_redocking_exploration.ipynb`'s LDA-vs-Rosetta-energy scatter): for
EVERY position Rosetta actually finds GA1 in contact with -- not just the
35 CDD-annotated ones -- how cleanly does that redocked-pose energetics
separate GA1 importers from non-importers, and does it line up with
NPF_LDA_kernel's own sequence-only importance where that even exists?

**2026-08-26, at the user's request**: restricting this scan to the CDD
putative binding site would silently ignore any real importer/non-importer
signal sitting on a residue CDD/InterPro's own domain model didn't flag.
Defaults to `redocking/results/rescoring/position_energetics_full.csv`
(all 746 whole-alignment positions, `build_position_mapping.py --full`'s
mapping -- reuses the SAME already-validated sequence alignment the 35
CDD positions come from, not a new structural TM-helix realignment; see
that script's own --full-mode caveat on why a position outside the pocket
is a lead worth checking against the real 3D structure, not as solid a
hit as a CDD-position one). `is_cdd_pocket` is carried through on every
row so the report always shows which kind of hit it is. Pass
`--cdd-only` to fall back to the original 35-position-only scan
(`lda_unfavorable_contacts.csv`) for a direct before/after comparison.

Reads rescore_redocked_aggregate.py's own output directly -- one row per
(protein, role, position) actually contacted in >=1 redocked complex, with
that protein's mean ligand<->residue twobody_total at that position -- no
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

REDOCKING_RESCORING_DIR = config.PIPELINE_ROOT / "redocking" / "results" / "rescoring"
FULL_ENERGETICS_CSV = REDOCKING_RESCORING_DIR / "position_energetics_full.csv"
CDD_ONLY_CSV = REDOCKING_RESCORING_DIR / "lda_unfavorable_contacts.csv"
OUT_CSV = REDOCKING_RESCORING_DIR / "position_cohesion_scan.csv"
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

        # "position" here can be the 1-746 whole-alignment index (--full mode) -- the
        # LDA Z-scale lookup always needs the 1-35 CDD numbering instead, carried
        # through as "cdd_position" (NaN outside the pocket) when that column exists;
        # CDD-only mode has no separate column, and there "position" already is 1-35.
        cdd_position = group["cdd_position"].iloc[0] if "cdd_position" in group.columns else position
        z_name, z_coef = dominant_z_scale(loadings, cdd_position) if pd.notna(cdd_position) else ("?", float("nan"))
        row = dict(
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
        )
        if "is_cdd_pocket" in group.columns:
            row["is_cdd_pocket"] = bool(group["is_cdd_pocket"].iloc[0])
            row["cdd_position"] = cdd_position
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mannwhitney_p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdd-only", action="store_true",
                     help="scan only the 35 CDD pocket positions (lda_unfavorable_contacts.csv), "
                          "the original narrower scope -- default now scans all 746 whole-alignment "
                          "positions (position_energetics_full.csv)")
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

    if args.cdd_only:
        energetics_csv = CDD_ONLY_CSV
        n_total_positions = "35"
    else:
        if not FULL_ENERGETICS_CSV.exists():
            raise SystemExit(f"{FULL_ENERGETICS_CSV} not found -- run "
                              "build_position_mapping.py --full then rescore_redocked_aggregate.py first, "
                              "or pass --cdd-only for the narrower 35-position scan.")
        energetics_csv = FULL_ENERGETICS_CSV
        n_total_positions = "746"

    energetics = pd.read_csv(energetics_csv)
    loadings = pd.read_csv(LDA_LOADINGS_TSV, sep="\t")

    result = scan(energetics, loadings, args.min_importer_n, args.min_non_importer_n)
    result.to_csv(OUT_CSV, index=False)
    n_cdd = int(result["is_cdd_pocket"].sum()) if "is_cdd_pocket" in result.columns else len(result)
    print(f"[scan_position_cohesion] wrote {OUT_CSV} ({len(result)}/{n_total_positions} positions cleared "
          f"the coverage minimums" + (f", {n_cdd} inside the CDD pocket, {len(result) - n_cdd} outside)" if "is_cdd_pocket" in result.columns else ")"))
    print("[scan_position_cohesion] CAVEAT: importer n<=10 per position -- Mann-Whitney p-values here are a "
          "ranking heuristic across positions, not multiple-testing-corrected significance claims. Positions "
          "outside the CDD pocket additionally rely on a sequence-only (not structural) alignment there -- see "
          "build_position_mapping.py --full's own caveat. Treat this as a shortlist to prioritize experimental "
          "follow-up, not a finished result.\n")

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
    if "is_cdd_pocket" in result.columns:
        cols += ["is_cdd_pocket", "cdd_position"]
    print(shortlist[cols].to_string(index=False))

    print("\n[scan_position_cohesion] full ranking (top 15 by p-value):")
    print(result[cols + ["direction"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
