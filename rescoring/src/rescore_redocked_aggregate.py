#!/usr/bin/env python3
"""
rescoring/src/rescore_redocked_aggregate.py
================================================
Pools rescore_redocked_batch.py's per-complex Rosetta ligand<->residue
decompositions and answers the two questions the redocked poses exist to
answer:

1. **Does physics-based redocking agree with the CDD/InterPro putative
   binding site better than ab initio cofolding did?** Same precision/
   recall framework plip_analysis.py's `cdd_agreement()` already uses for
   PLIP-vs-CDD on ab initio poses (rescoring/results/plip_cdd_agreement.csv),
   applied here to Rosetta's own energy-graph contacts (a residue counts as
   "interacting"/"energy-relevant" iff decompose.py found a non-zero
   ligand<->residue energy-graph edge for it -- geometric contact and
   energetic relevance are the same test here, not two separate cutoffs) on
   the HADDOCK3-redocked poses, restricted to the SAME method (Rosetta
   two-body decomposition, not PLIP) on the SAME 5 GA1-importer proteins
   ab initio holoform poses exist for, so the before/after comparison is
   apples-to-apples. `rescoring/results/plip_cdd_agreement.csv`'s existing
   PLIP-based numbers are also reported alongside for reference (the
   figure the user recalled as "~50%": pooled recall = 0.480 across the 8
   GA1-associated proteins there, precision = 0.348).

2. **Importer vs. non-importer, on the redocked poses themselves** (mirrors
   NPF_LDA_kernel's own importer/non-importer framing): no ab initio GA1
   pose exists for non-importers to compare against (they were only ever
   co-folded with their OWN assigned ligand or apoform) -- redocking is
   what gives non-importers a physically-derived GA1-pose hypothesis at
   all. Pooled precision/recall computed separately per role answers "does
   HADDOCK3 converge on a pocket-engaging pose in importers but not (or
   less so) in non-importers, independent of what ABCfold predicted?"

3. **Unfavorable (positive REU) contact at the positions NPF_LDA_kernel's
   sequence classifier flags as most important** -- for each CDD pocket
   position actually contacted in at least one redocked complex, the mean
   ligand<->residue twobody_total energy next to that position's LDA
   importance (data/position_importance_GA1.tsv). A position with HIGH LDA
   importance but a POSITIVE (repulsive/unfavorable) mean energy in the
   redocked poses is the interesting/concerning case -- a residue the
   sequence-only classifier thinks matters for binding, but that GA1's own
   physically-docked pose actually clashes with rather than favorably
   contacts.

Run after rescore_redocked_batch.py has produced at least some
redocking/results/rescoring/per_complex/*.csv.

Output (all under redocking/results/rescoring/):
  all_contacts.csv              pooled long table, position-mapped (35 CDD positions only)
  cdd_agreement.csv             per (protein, role): precision/recall vs. CDD
  lda_unfavorable_contacts.csv  per (protein, position): mean energy vs. LDA importance (CDD positions only)
  position_energetics_full.csv  same as above, but every one of the 746 whole-alignment
                                 positions (build_position_mapping.py --full), with an
                                 is_cdd_pocket column -- 2026-08-26, at the user's request:
                                 a Rosetta-contacted residue outside the CDD-annotated
                                 pocket is still a real signal, not silently dropped.
                                 lda_importance is NaN outside the 35 CDD positions (the
                                 classifier was never fit there).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import config

REDOCKING_ROOT = config.PIPELINE_ROOT / "redocking"
DEFAULT_PER_COMPLEX_DIR = REDOCKING_ROOT / "results" / "rescoring" / "per_complex"
DEFAULT_OUT_DIR = REDOCKING_ROOT / "results" / "rescoring"

# The 5 GA1-importer proteins that have BOTH a redocked complex here AND a
# real ABCfold GA1-holoform pose already scored in rescoring/results/ --
# see redocking/README.md's "Scope" section. Fixed here (not re-derived)
# since it's the exact set the apples-to-apples ab-initio-baseline
# recomputation below needs to match.
AB_INITIO_GA1_PROTEINS = [
    "NPF3.1_Q9SX20", "NPF2.12_Q9LFX9", "NPF2.13_Q8RX77", "NPF2.10_Q944G5", "NPF2.5_Q9M172",
]


def load_all_contacts(per_complex_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(per_complex_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        sys.exit(f"No per-complex results found in {per_complex_dir} -- run rescore_redocked_batch.py first.")
    return pd.concat(frames, ignore_index=True)


def add_position(contacts: pd.DataFrame, position_map_csv: Path = config.POSITION_RESNR_MAP_CSV) -> pd.DataFrame:
    """Merge in `position` (and `is_cdd_pocket`, when using the --full map --
    see build_position_mapping.py) for every contact row. Defaults to the
    35-CDD-position-only map (unchanged behavior); pass
    config.POSITION_RESNR_MAP_FULL_CSV for the whole-alignment (746-column)
    version instead -- see main()'s two separate calls."""
    position_map = pd.read_csv(position_map_csv)
    merged = contacts.merge(
        position_map, left_on=["protein", "prot_resi"], right_on=["protein", "resnr"], how="left",
    )
    n_unmapped = merged["position"].isna().sum()
    if n_unmapped:
        scope = "the CDD-mapped alignment" if position_map_csv == config.POSITION_RESNR_MAP_FULL_CSV else "the 35 CDD pocket positions"
        print(f"[rescore_redocked_aggregate] NOTE: {n_unmapped}/{len(merged)} contact rows are outside "
              f"{scope} -- kept with position=NaN, dropped from the position-indexed aggregates below.")
    return merged


def cdd_agreement(contacts_with_position: pd.DataFrame) -> pd.DataFrame:
    """Per protein: of the residues Rosetta's own energy graph found
    actually interacting with GA1 in the redocked pose(s) (pooled across
    that protein's ca_cluster complexes), how many fall inside vs. outside
    the 35 CDD pocket positions (precision), and of the 35, how many are
    ever contacted (recall)? Same formula as plip_analysis.cdd_agreement,
    Rosetta contacts instead of PLIP ones."""
    position_map = pd.read_csv(config.POSITION_RESNR_MAP_CSV)

    rows = []
    for (protein, role), group in contacts_with_position.groupby(["protein", "role"]):
        unique_residues = group.drop_duplicates(["prot_resi"])
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
            protein=protein, role=role, n_complexes=group["complex_id"].nunique(),
            n_unique_contacted_residues=n_contacted,
            n_in_cdd_pocket=n_in_cdd, n_outside_cdd_pocket=n_out_cdd, precision=precision,
            n_cdd_positions_total=n_cdd_total, n_cdd_positions_contacted=n_cdd_contacted, recall=recall,
        ))
    return pd.DataFrame(rows).sort_values(["role", "protein"])


def _pooled(agreement: pd.DataFrame, role: str | None = None) -> tuple[float, float]:
    sub = agreement if role is None else agreement[agreement["role"] == role]
    precision = sub["n_in_cdd_pocket"].sum() / sub["n_unique_contacted_residues"].sum()
    recall = sub["n_cdd_positions_contacted"].sum() / sub["n_cdd_positions_total"].sum()
    return precision, recall


def lda_unfavorable(contacts_with_position: pd.DataFrame) -> pd.DataFrame:
    """Per (protein, position) actually contacted in >=1 redocked complex:
    mean ligand<->residue twobody_total (this position's net Rosetta
    energetic contribution, averaged across that protein's replicas/
    ca_clusters where contacted) next to NPF_LDA_kernel's sequence-derived
    importance for that position (NaN outside the 35 CDD positions -- the
    classifier was only ever fit there, see build_position_mapping.py's
    --full-mode caveat). `unfavorable` = mean energy > 0 (net repulsive/
    destabilizing). Works on either the CDD-only or --full position map
    (whichever `add_position()` call produced `contacts_with_position`) --
    carries `is_cdd_pocket` through when that column is present, so a
    --full-mode caller can tell pocket hits from outside-pocket ones."""
    importance_path = config.DATA_DIR / "position_importance_GA1.tsv"
    importance = pd.read_csv(importance_path, sep="\t")

    mapped = contacts_with_position.dropna(subset=["position"])
    # one twobody_total per (complex_id, position) -- dedupe across scoretype rows
    per_edge = mapped.drop_duplicates(["complex_id", "position"])
    has_cdd_flag = "is_cdd_pocket" in per_edge.columns
    # --full mode's "position" is a 1-746 whole-alignment index, a DIFFERENT numbering
    # scheme than lda_GA1_loadings.tsv/position_importance_GA1.tsv's own 1-35 CDD
    # "position" column -- the LDA-importance merge below must use "cdd_position"
    # (build_position_mapping.py --full's own 1-35-numbered column, NaN outside the
    # pocket) when present, not "position" itself. CDD-only mode has no separate
    # cdd_position column -- there, "position" already IS the 1-35 numbering.
    merge_key = "cdd_position" if "cdd_position" in per_edge.columns else "position"

    rows = []
    for (protein, role, position), group in per_edge.groupby(["protein", "role", "position"]):
        mean_energy = group["twobody_total"].mean()
        row = dict(
            protein=protein, role=role, position=position,
            mean_twobody_total=mean_energy, n=len(group),
            resnr_examples=sorted(group["prot_resi"].unique().tolist())[:3],
        )
        if has_cdd_flag:
            row["is_cdd_pocket"] = bool(group["is_cdd_pocket"].iloc[0])
            row["cdd_position"] = group["cdd_position"].iloc[0] if "cdd_position" in group.columns else None
        rows.append(row)
    result = pd.DataFrame(rows).merge(
        importance.rename(columns={"position": merge_key}), on=merge_key, how="left",
    )
    result = result.rename(columns={"importance": "lda_importance"})
    result["unfavorable"] = result["mean_twobody_total"] > 0
    return result.sort_values(["role", "protein", "lda_importance"], ascending=[True, True, False])


def ab_initio_rosetta_baseline() -> pd.DataFrame | None:
    """Recompute the SAME precision/recall on rescoring/results/all_contacts.csv
    (this project's own ab-initio-pose Rosetta decomposition), restricted to
    GA1 rows for AB_INITIO_GA1_PROTEINS -- the fair, same-method "before"
    number for the importer-side comparison. None if that file doesn't
    exist yet (rescoring's own aggregate.py hasn't been run)."""
    ab_initio_path = config.RESULTS_DIR / "all_contacts.csv"
    if not ab_initio_path.exists():
        print(f"[rescore_redocked_aggregate] {ab_initio_path} not found -- run rescoring/src/aggregate.py "
              "first for the ab-initio Rosetta baseline comparison. Skipping.")
        return None
    contacts = pd.read_csv(ab_initio_path)
    contacts = contacts[(contacts["ligand"] == "GA1") & (contacts["protein"].isin(AB_INITIO_GA1_PROTEINS))]
    if contacts.empty:
        print("[rescore_redocked_aggregate] no GA1 ab-initio rows found for the importer protein set -- skipping baseline.")
        return None
    contacts = contacts.rename(columns={"complex_id": "complex_id"})
    contacts["role"] = "importer_ab_initio"
    return cdd_agreement(contacts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-complex-dir", type=Path, default=DEFAULT_PER_COMPLEX_DIR,
                     help="rescore_redocked_batch.py's --out-dir (default: results/rescoring/per_complex)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                     help="where to write all_contacts/cdd_agreement/lda_unfavorable_contacts.csv "
                          "(default: results/rescoring)")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_contacts = load_all_contacts(args.per_complex_dir)
    print(f"[rescore_redocked_aggregate] loaded {len(raw_contacts)} rows from "
          f"{raw_contacts['complex_id'].nunique()} redocked complexes "
          f"({raw_contacts[raw_contacts['role'] == 'importer']['complex_id'].nunique()} importer, "
          f"{raw_contacts[raw_contacts['role'] == 'non_importer']['complex_id'].nunique()} non_importer)")
    contacts = add_position(raw_contacts)
    contacts.to_csv(out_dir / "all_contacts.csv", index=False)

    agreement = cdd_agreement(contacts)
    agreement.to_csv(out_dir / "cdd_agreement.csv", index=False)
    print(f"[rescore_redocked_aggregate] wrote cdd_agreement.csv ({len(agreement)} protein rows)")

    for role in ["importer", "non_importer"]:
        if (agreement["role"] == role).any():
            precision, recall = _pooled(agreement, role)
            print(f"[rescore_redocked_aggregate] REDOCKED {role}: pooled precision={precision:.3f}, "
                  f"pooled recall={recall:.3f}")

    baseline = ab_initio_rosetta_baseline()
    if baseline is not None:
        base_precision, base_recall = _pooled(baseline)
        redocked_importer = agreement[agreement["role"] == "importer"]
        if not redocked_importer.empty:
            redocked_precision, redocked_recall = _pooled(redocked_importer)
            print(f"[rescore_redocked_aggregate] AB-INITIO importer baseline (Rosetta, same {len(AB_INITIO_GA1_PROTEINS)} "
                  f"proteins): pooled precision={base_precision:.3f}, pooled recall={base_recall:.3f}")
            print(f"[rescore_redocked_aggregate] REDOCKED importer improvement: "
                  f"precision {base_precision:.3f} -> {redocked_precision:.3f}, "
                  f"recall {base_recall:.3f} -> {redocked_recall:.3f}")
        baseline.to_csv(out_dir / "ab_initio_rosetta_baseline_cdd_agreement.csv", index=False)

    plip_path = config.RESULTS_DIR / "plip_cdd_agreement.csv"
    if plip_path.exists():
        plip = pd.read_csv(plip_path)
        ga1 = plip[plip["ligand"] == "GA1"]
        if not ga1.empty:
            plip_precision = ga1["n_in_cdd_pocket"].sum() / ga1["n_unique_contacted_residues"].sum()
            plip_recall = ga1["n_cdd_positions_contacted"].sum() / ga1["n_cdd_positions_total"].sum()
            print(f"[rescore_redocked_aggregate] (for reference) ab-initio PLIP-based GA1 agreement "
                  f"(the ~50% figure): pooled precision={plip_precision:.3f}, pooled recall={plip_recall:.3f}")

    unfavorable = lda_unfavorable(contacts)
    unfavorable.to_csv(out_dir / "lda_unfavorable_contacts.csv", index=False)
    n_unfav = int(unfavorable["unfavorable"].sum())
    print(f"[rescore_redocked_aggregate] wrote lda_unfavorable_contacts.csv "
          f"({len(unfavorable)} (protein, position) rows, {n_unfav} flagged unfavorable)")

    # Whole-alignment (746-column) version, at the user's request (2026-08-26):
    # a Rosetta-contacted residue outside the 35 CDD positions is still a real
    # signal -- see build_position_mapping.py's --full mode and its own caveat
    # (sequence, not structural, alignment outside the pocket). cdd_agreement()
    # deliberately still only uses the CDD-only `contacts` above -- precision/
    # recall vs. the CDD pocket is a CDD-specific question by definition.
    if config.POSITION_RESNR_MAP_FULL_CSV.exists():
        contacts_full = add_position(raw_contacts, config.POSITION_RESNR_MAP_FULL_CSV)
        unfavorable_full = lda_unfavorable(contacts_full)
        unfavorable_full.to_csv(out_dir / "position_energetics_full.csv", index=False)
        n_outside_cdd = int((~unfavorable_full["is_cdd_pocket"]).sum())
        print(f"[rescore_redocked_aggregate] wrote position_energetics_full.csv "
              f"({len(unfavorable_full)} (protein, position) rows across all 746 alignment columns, "
              f"{n_outside_cdd} outside the 35 CDD positions)")
    else:
        print(f"[rescore_redocked_aggregate] {config.POSITION_RESNR_MAP_FULL_CSV} not found -- run "
              "build_position_mapping.py --full first for the whole-alignment position scan.")


if __name__ == "__main__":
    main()
