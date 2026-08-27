"""
mmgbsa/src/compare_engines.py
=============================
Stage 7: put the MD-averaged GB per-residue decomposition next to the
single-pose PyRosetta REF2015 two-body decomposition on the SAME redocked
poses, mapped onto the SAME alignment positions.

Merges on (protein, position):
  GB       results/mmgbsa/decomp_by_position.csv          -> mean_gb_total
  Rosetta  redocking/results/rescoring/position_energetics_full.csv -> mean_twobody_total

Reports:
  * Spearman rho (GB vs Rosetta per-position mean) overall and per role
  * for the Rosetta scan's shortlist positions (CDD 15, 1, 20 by default):
    does GB agree on sign / on importer-favorable direction?
  * per-complex: dG_GB (binding_energy_summary.csv) vs HADDOCK score
    (data/manifest.csv) -- rank correlation, importer vs non_importer means

Output: results/mmgbsa/engine_comparison.csv + printed summary.

    python compare_engines.py [--shortlist 15,1,20]
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import spearmanr

import config


def _per_position_spearman(merged: pd.DataFrame, label: str) -> None:
    sub = merged.dropna(subset=["mean_gb_total", "mean_twobody_total"])
    if len(sub) < 3:
        print(f"[stage7]   {label:<14} n={len(sub)} (too few to correlate)")
        return
    rho, p = spearmanr(sub["mean_gb_total"], sub["mean_twobody_total"])
    print(f"[stage7]   {label:<14} n={len(sub):<5} Spearman rho={rho:+.3f}  p={p:.2g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", default="15,1,20", help="CDD positions to check agreement on")
    args = ap.parse_args()

    if not config.DECOMP_BY_POSITION_CSV.exists():
        raise SystemExit(f"{config.DECOMP_BY_POSITION_CSV} not found -- run aggregate_decomp.py first.")
    gb = pd.read_csv(config.DECOMP_BY_POSITION_CSV)

    if not config.ROSETTA_POSITION_ENERGETICS_CSV.exists():
        raise SystemExit(f"{config.ROSETTA_POSITION_ENERGETICS_CSV} not found -- run redocking/ "
                         "rescore_redocked_aggregate.py first.")
    ros = pd.read_csv(config.ROSETTA_POSITION_ENERGETICS_CSV)

    merged = gb.merge(
        ros[["protein", "role", "position", "mean_twobody_total", "cdd_position"]],
        on=["protein", "role", "position"], how="outer", suffixes=("", "_ros"),
    )
    merged["cdd_position"] = merged["cdd_position"].fillna(merged["cdd_position_ros"])
    merged.to_csv(config.MMGBSA_DIR / "engine_comparison.csv", index=False)

    print("[stage7] per-position GB vs Rosetta two-body energy (Spearman):")
    _per_position_spearman(merged, "overall")
    for role, sub in merged.groupby("role"):
        _per_position_spearman(sub, role)

    print("\n[stage7] shortlist-position agreement (GB sign vs Rosetta sign, importer rows):")
    targets = [int(x) for x in args.shortlist.split(",") if x.strip()]
    for t in targets:
        rowset = merged[(merged["cdd_position"] == t) & (merged["role"] == "importer")]
        if rowset.empty:
            print(f"[stage7]   CDD {t:<3} -- no importer rows in the merge")
            continue
        gb_mean = rowset["mean_gb_total"].mean()
        ros_mean = rowset["mean_twobody_total"].mean()
        agree = "agree" if (gb_mean < 0) == (ros_mean < 0) else "DISAGREE"
        print(f"[stage7]   CDD {t:<3} GB={gb_mean:+.3f}  Rosetta={ros_mean:+.3f}  -> {agree} on sign")

    # per-complex dG_GB vs HADDOCK score
    if config.BINDING_ENERGY_SUMMARY_CSV.exists():
        be = pd.read_csv(config.BINDING_ENERGY_SUMMARY_CSV)
        man = pd.DataFrame(config.read_csv_rows(config.MANIFEST_CSV))
        man["haddock_score"] = pd.to_numeric(man["haddock_score"], errors="coerce")
        perc = (be.groupby(["complex_id", "protein", "role"])["dg_gb_total"].mean()
                .reset_index().merge(man[["complex_id", "haddock_score"]], on="complex_id"))
        print("\n[stage7] per-complex dG_GB (mean over replicas):")
        for role, sub in perc.groupby("role"):
            print(f"[stage7]   {role:<13} mean dG_GB={sub['dg_gb_total'].mean():+8.2f} kcal/mol  (n={len(sub)})")
        sub = perc.dropna(subset=["dg_gb_total", "haddock_score"])
        if len(sub) >= 3:
            rho, p = spearmanr(sub["dg_gb_total"], sub["haddock_score"])
            print(f"[stage7]   dG_GB vs HADDOCK score: Spearman rho={rho:+.3f}  p={p:.2g}  n={len(sub)}")

    print(f"\n[stage7] wrote {config.MMGBSA_DIR / 'engine_comparison.csv'}")


if __name__ == "__main__":
    main()
