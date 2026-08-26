"""
redocking/src/make_manifest.py
=================================
Stage 2: this run's receptor complexes -- one structure per (protein,
macro-conformation ca_cluster) for every CDD-annotated protein in the
corpus (48 of 53 -- the 5 without CDD pocket residues at all are the only
ones excluded outright). Per-protein pca_k3 clustering typically yields 3
macro-conformations (ca_cluster 0/1/2) -- rather than picking a single
"primary" cluster's pose, every usable macro-conformation gets its own
docking run, since different backbone conformations (e.g. inward- vs
outward-facing transporter states) can have meaningfully different pocket
accessibility -- confirmed by hand that all 48 candidate proteins have
exactly 3 such clusters, giving 144 total complexes.

**2026-08-26: extended from the original 24-protein scope** (NPF_LDA_kernel's
curated `hc_importers`/`hc_non_importers` lists only, 72 complexes) to the
full CDD-annotated corpus, at the user's request -- more non_importer-side
data means more statistical power for `rescoring/src/scan_position_cohesion.py`'s
importer-vs-non-importer position scan. Every CDD-annotated protein gets a
role via `_classify_role()`:
  - **importer**: in `hc_importers` AND actually co-folded with GA1 here
    (`ligand_for() == "GA1"` -- see the note on that function below). Still
    exactly 5 (unchanged from the original scope).
  - **non_importer**: either in `hc_non_importers` (19, unchanged), OR any
    other CDD-annotated protein NOT co-folded with GA1 here (confirmed
    co-folded with something else -- nitrate/ABA/JA-Ile/quercetin-3-O-
    sophoroside -- or apoform only) -- 22 new proteins, all using the SAME
    apoform receptor status as the original 19, for a consistent,
    ligand-choice-unbiased receptor conformation across the whole
    non_importer population regardless of what ligand (if any) a given
    protein happens to be assigned elsewhere in this pipeline.
  - **ambiguous**: CDD-annotated, actually co-folded with GA1 here, but
    NOT in NPF_LDA_kernel's curated `hc_importers` list -- 2 proteins
    (`NPF2.1_Q9M171`, `NPF5.6_P0CI03`) ABCfold happened to test with GA1
    without a curated confirmed-importer label backing that choice. Per
    the user (2026-08-26): redock them (their real ABCfold GA1 pose is
    already there for a Stage-7 RMSD comparison, same as importers get),
    but exclude them from the importer-vs-non-importer statistics
    everywhere downstream until their status is actually confirmed --
    `compare_to_abcfold.py`/`rescore_redocked_aggregate.py`/
    `scan_position_cohesion.py` only ever loop over
    `["importer", "non_importer"]` explicitly, so `ambiguous` rows are
    already excluded from every pooled statistic without any extra code
    -- just don't add "ambiguous" to those loops.

Receptor source: results/tm_reannotated/<protein>/pca_k3/assignments.parquet
+ .../cluster_<id>/<frame_id>.cif (worflows/postprocessing's own TM-
alignment + PCA-k3 macro-state clustering, scripts/cluster_conformations.py
-- pools apo AND holo frames on one shared coordinate frame). Used instead
of raw results/abcfold/ or rescoring/data/manifest.csv (the previous,
smaller-scale source) because a meaningful fraction of raw per-frame CIFs
get deleted by this pipeline's own storage-compression step once run on
the cluster (confirmed by hand: 16 of the original 24 manifest rows'
raw CIFs were already gone on the IFB cluster, still present locally) --
tm_reannotated's `symlinked == True` frames are a separately curated,
stable subset that survives that compression.

Per-(protein, ca_cluster) representative pick: among symlinked frames in
that cluster, highest ptm (overall fold confidence -- the receptor's own
quality, not iptm's ligand-placement confidence, which is irrelevant here
since the ab initio ligand pose is discarded and redocked fresh), frame_id
as a deterministic tiebreak.

Output: data/manifest.csv (complex_id, protein, role [importer/
non_importer/ambiguous], form [holo/apo], ca_cluster, receptor_cif) + a
coverage report printed to stdout (skipped proteins + why).
"""
from __future__ import annotations

import csv
import hashlib
import sys

import config

# ligand_for() is the single source of truth for which ligand a protein's
# "__holo" ABCfold run was actually co-folded with -- NOT necessarily GA1.
# Confirmed by hand (2026-08-25): naively accepting any status=="holo"
# tm_reannotated frame for an hc_importers-list protein silently pulled in
# NPF2.7/NPF2.3/NPF2.4/NPF1.1/NPF1.2's NITRATE-bound holoform conformations
# as if they were GA1 importers (they're in NITRATE_TRANSPORTERS, not
# HC_IMPORTERS, in ligand_assignment.py) -- those receptor conformations
# were never challenged with GA1 during ab initio prediction, and there's
# no GA1 ABCfold pose to compare against for them anyway (the same reason
# they were excluded from the original, smaller-scale manifest).
sys.path.insert(0, str(config.PIPELINE_ROOT / "scripts"))
from ligand_assignment import ligand_for  # noqa: E402


def _classify_role(short: str, importers: set[str], non_importers: set[str]) -> tuple[str, str]:
    """(role, receptor status) for one protein, given its short name (e.g.
    'NPF2.1') -- see module docstring for the 3-way importer/non_importer/
    ambiguous split.

    Per the user (2026-08-26): a curated `hc_importers` protein this
    pipeline happened to co-fold with a DIFFERENT ligand (nitrate, for
    NPF1.1/NPF1.2/NPF2.3/NPF2.4/NPF2.7 -- real biology, not a data error:
    several NPF transporters are dual/low-affinity GA1 importers on top of
    their primary nitrate role) is still a real importer, just without an
    ABCfold GA1 pose to compare against -- do NOT skip it. Use the
    ligand-unbiased apoform receptor for these (same choice non_importer
    already uses) instead of their nitrate-bound holoform, and let
    compare_to_abcfold.py fall back to the non-importer-style pocket-
    engagement comparison for them (no RMSD-vs-ABCfold-pose is possible
    without a real GA1 pose) while still counting them as `role=importer`
    everywhere else (rescore_redocked_aggregate.py's CDD agreement,
    scan_position_cohesion.py's Mann-Whitney test, etc.) -- exactly the
    additional importer-side statistical power this expansion exists for."""
    lig = ligand_for(short)
    if short in importers:
        return ("importer", "holo") if lig == config.LIGAND_KEY else ("importer", "apo")
    if short in non_importers:
        return "non_importer", "apo"
    if lig == config.LIGAND_KEY:
        return "ambiguous", "holo"
    return "non_importer", "apo"


def _cluster_representatives(protein: str, status: str) -> list[dict] | None:
    """One row per usable ca_cluster for `protein`'s `status` ('apo' or
    'holo') frames -- None if this protein has no clustering output at
    all, empty list if it has output but nothing usable for this status."""
    df = config.load_cluster_assignments(protein)
    if df is None:
        return None
    sub = df[(df["status"] == status) & (df["symlinked"])]
    if sub.empty:
        return []

    reps = []
    for cluster_id, group in sub.groupby("cluster"):
        best = group.sort_values(["ptm", "frame_id"], ascending=[False, True]).iloc[0]
        reps.append({"cluster": int(cluster_id), "frame_id": best["frame_id"],
                      "model": best["model"], "ptm": float(best["ptm"])})
    return reps


def _complex_id(protein: str, cluster: int, model: str, frame_id: str) -> str:
    """Short and unique regardless of backend -- some backends' frame_id
    (e.g. boltz) embeds the whole nested output directory structure and
    is 200+ characters on its own (confirmed by hand on the earlier,
    single-cluster-per-protein manifest); a short hash of the real
    frame_id keeps this collision-free without the unwieldy length."""
    short_hash = hashlib.md5(frame_id.encode()).hexdigest()[:8]
    return f"{protein}__ca{cluster}_{model}_{short_hash}"


def build_manifest() -> tuple[list[dict], list[str]]:
    manifest_rows: list[dict] = []
    skipped: list[str] = []

    importers = set(config.load_importers())
    non_importers = set(config.load_non_importers())

    for full in config.cdd_annotated_proteins():
        short = full.split("_")[0]
        role, status = _classify_role(short, importers, non_importers)
        reps = _cluster_representatives(full, status)
        if reps is None:
            skipped.append(f"{full}: no assignments.parquet found")
            continue
        if not reps:
            skipped.append(f"{full}: assignments.parquet exists but no symlinked {status!r} frames in it")
            continue
        for rep in reps:
            cif_path = config.cluster_cif_path(full, rep["cluster"], rep["frame_id"])
            if not cif_path.exists():
                skipped.append(f"{full} ca_cluster={rep['cluster']}: {cif_path} listed as symlinked "
                                f"but file not found on disk")
                continue
            complex_id = _complex_id(full, rep["cluster"], rep["model"], rep["frame_id"])
            manifest_rows.append({
                "complex_id": complex_id, "protein": full, "role": role, "form": status,
                "ca_cluster": rep["cluster"],
                "receptor_cif": str(cif_path.relative_to(config.PIPELINE_ROOT)),
            })

    return manifest_rows, skipped


def main() -> None:
    rows, skipped = build_manifest()
    with config.MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["complex_id", "protein", "role", "form", "ca_cluster", "receptor_cif"])
        writer.writeheader()
        writer.writerows(rows)

    n_importer = sum(1 for r in rows if r["role"] == "importer")
    n_non_importer = sum(1 for r in rows if r["role"] == "non_importer")
    n_ambiguous = sum(1 for r in rows if r["role"] == "ambiguous")
    print(f"Wrote {len(rows)} rows to {config.MANIFEST_CSV} "
          f"({n_importer} importer, {n_non_importer} non_importer, {n_ambiguous} ambiguous)")
    for r in rows:
        print(f"  {r['role']:>13s}  {r['protein']:<16s}  ca{r['ca_cluster']}  {r['receptor_cif']}")

    if skipped:
        print(f"\nSkipped {len(skipped)} candidate(s):")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
