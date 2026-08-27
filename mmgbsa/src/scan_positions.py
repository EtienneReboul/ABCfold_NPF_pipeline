"""
mmgbsa/src/scan_positions.py
============================
Stage 6: the importer-vs-non_importer per-position scan, exactly the design of
rescoring/src/scan_position_cohesion.py but on the MD-averaged GB per-residue
decomposition (Stage 5's decomp_by_position.csv) instead of the single-pose
PyRosetta two-body energy.

For every alignment position with enough proteins contacted on both sides:
  * frac_unfavorable per role  (fraction of that role's proteins whose
    mean_gb_total at the position is > 0 -- repulsive/desolvation-dominated)
  * Mann-Whitney U (two-sided) on mean_gb_total, importer vs non_importer
  * Benjamini-Hochberg FDR over the positions that actually cleared the
    coverage minimums (scipy.stats.false_discovery_control, method="bh")
  * dominant NPF_LDA_kernel Z-scale driver at the position (from
    rescoring/data/lda_GA1_loadings.tsv, keyed by cdd_position)

Headline question: does the MD-averaged GB decomposition independently
recover CDD position 15 (and the weaker secondaries 1 and 20) that the
Rosetta scan flagged (redocking/RESULTS.md sec5)?

Output: results/mmgbsa/position_cohesion_scan_gbsa.csv + a printed shortlist.

    python scan_positions.py [--cdd-only] [--min-importer-n N]
        [--min-non-importer-n N] [--q-threshold Q] [--min-gap G]
        [--max-importer-unfavorable-frac F]
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

import config


def scan(df: pd.DataFrame, dom_z: dict[int, str], min_imp: int, min_non: int) -> pd.DataFrame:
    df = df.copy()
    df["unfavorable"] = df["mean_gb_total"] > 0
    rows = []
    for position, g in df.groupby("position"):
        imp = g[g["role"] == "importer"]
        non = g[g["role"] == "non_importer"]
        if len(imp) < min_imp or len(non) < min_non:
            continue
        try:
            _, p = mannwhitneyu(imp["mean_gb_total"], non["mean_gb_total"], alternative="two-sided")
        except ValueError:
            p = float("nan")
        mean_imp = imp["mean_gb_total"].mean()
        mean_non = non["mean_gb_total"].mean()
        cdd_pos = g["cdd_position"].dropna().iloc[0] if g["cdd_position"].notna().any() else None
        rows.append(dict(
            position=int(position),
            is_cdd_pocket=bool(g["is_cdd_pocket"].iloc[0]),
            cdd_position=cdd_pos,
            n_importer=len(imp), n_non_importer=len(non),
            frac_unfavorable_importer=imp["unfavorable"].mean(),
            frac_unfavorable_non_importer=non["unfavorable"].mean(),
            unfavorable_gap=non["unfavorable"].mean() - imp["unfavorable"].mean(),
            mean_gb_importer=mean_imp, mean_gb_non_importer=mean_non,
            delta_mean=mean_non - mean_imp,
            mannwhitney_p=p,
            direction="importer_favorable" if mean_imp < mean_non else "non_importer_favorable",
            dominant_z_scale=dom_z.get(int(cdd_pos), "?") if cdd_pos is not None else "?",
        ))
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    valid = res["mannwhitney_p"].notna()
    res["bh_qvalue"] = float("nan")
    if valid.any():
        res.loc[valid, "bh_qvalue"] = false_discovery_control(
            res.loc[valid, "mannwhitney_p"].to_numpy(), method="bh")
    return res.sort_values("mannwhitney_p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdd-only", action="store_true",
                    help="restrict to the 35 CDD pocket positions (the sec5 family in redocking/RESULTS.md)")
    ap.add_argument("--min-importer-n", type=int, default=4)
    ap.add_argument("--min-non-importer-n", type=int, default=12)
    ap.add_argument("--q-threshold", type=float, default=0.2)
    ap.add_argument("--min-gap", type=float, default=0.3)
    ap.add_argument("--max-importer-unfavorable-frac", type=float, default=0.25)
    args = ap.parse_args()

    if not config.DECOMP_BY_POSITION_CSV.exists():
        raise SystemExit(f"{config.DECOMP_BY_POSITION_CSV} not found -- run aggregate_decomp.py first.")
    df = pd.read_csv(config.DECOMP_BY_POSITION_CSV)
    if args.cdd_only:
        df = df[df["is_cdd_pocket"].astype(str).str.lower().isin(("true", "1"))]

    dom_z = config.load_dominant_z()
    res = scan(df, dom_z, args.min_importer_n, args.min_non_importer_n)
    if res.empty:
        print("[stage6] no position cleared the coverage minimums "
              f"(min importer {args.min_importer_n}, min non_importer {args.min_non_importer_n}) -- "
              "expected on the smoke set (n=3); rerun after the full batch.")
        res.to_csv(config.POSITION_SCAN_CSV, index=False)
        return
    res.to_csv(config.POSITION_SCAN_CSV, index=False)

    n_cdd = int(res["is_cdd_pocket"].sum())
    print(f"[stage6] {len(res)} positions tested ({n_cdd} CDD, {len(res) - n_cdd} outside) "
          f"-> {config.POSITION_SCAN_CSV}")
    print("[stage6] BH-FDR over the tested family; raw Mann-Whitney p is a ranking heuristic only.\n")

    cols = ["position", "cdd_position", "is_cdd_pocket", "n_importer", "n_non_importer",
            "frac_unfavorable_importer", "frac_unfavorable_non_importer",
            "mean_gb_importer", "mean_gb_non_importer", "mannwhitney_p", "bh_qvalue",
            "direction", "dominant_z_scale"]
    shortlist = res[
        (res["bh_qvalue"] <= args.q_threshold)
        & (res["direction"] == "importer_favorable")
        & (res["unfavorable_gap"] >= args.min_gap)
        & (res["frac_unfavorable_importer"] <= args.max_importer_unfavorable_frac)
    ]
    print(f"[stage6] {len(shortlist)} position(s) in the follow-up shortlist "
          f"(BH q<={args.q_threshold}, importer-favorable, gap>={args.min_gap}):")
    print(shortlist[cols].to_string(index=False) if not shortlist.empty else "  (none)")
    print("\n[stage6] top 15 by p-value:")
    print(res[cols].head(15).to_string(index=False))

    for target in (15, 1, 20):
        hit = res[res["cdd_position"] == target]
        if not hit.empty:
            r = hit.iloc[0]
            print(f"[stage6] CDD position {target}: p={r['mannwhitney_p']:.4g} q={r['bh_qvalue']:.3g} "
                  f"dir={r['direction']}")
        else:
            print(f"[stage6] CDD position {target}: not tested (insufficient coverage in GB set)")


if __name__ == "__main__":
    main()
