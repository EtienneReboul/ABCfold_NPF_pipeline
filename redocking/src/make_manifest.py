"""
redocking/src/make_manifest.py
=================================
Stage 2: this run's receptor complexes -- one structure per (protein,
macro-conformation ca_cluster) for every NPF_LDA_kernel HC importer/non-
importer that has both CDD pocket residues and usable clustering output.
Per-protein pca_k3 clustering typically yields 3 macro-conformations
(ca_cluster 0/1/2) -- rather than picking a single "primary" cluster's
pose (this pipeline's earlier, smaller pilot), every usable macro-
conformation gets its own docking run, since different backbone
conformations (e.g. inward- vs outward-facing transporter states) can
have meaningfully different pocket accessibility -- confirmed by hand
(2026-08-25) that all 24 candidate proteins have exactly 3 such clusters,
giving 72 total complexes (5 importers x 3 + 19 non-importers x 3).

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
stable subset that survives that compression, and cover both apo and holo
frames identically, so importers and non-importers now share one
selection code path instead of two.

Per-(protein, ca_cluster) representative pick: among symlinked frames in
that cluster, highest ptm (overall fold confidence -- the receptor's own
quality, not iptm's ligand-placement confidence, which is irrelevant here
since the ab initio ligand pose is discarded and redocked fresh), frame_id
as a deterministic tiebreak.

Output: data/manifest.csv (complex_id, protein, role [importer/
non_importer], form [holo/apo], ca_cluster, receptor_cif) + a coverage
report printed to stdout (skipped proteins + why).
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


def _full_protein_name(short_name: str) -> str | None:
    """'NPF2.10' -> 'NPF2.10_Q944G5', by matching against whichever
    tm_reannotated protein directories exist -- avoids hardcoding UniProt
    accessions here. None if no match found."""
    if not config.TM_REANNOTATED_ROOT.exists():
        return None
    candidates = [d.name for d in config.TM_REANNOTATED_ROOT.iterdir() if d.name.startswith(short_name + "_")]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"{short_name!r} matched multiple full names: {sorted(candidates)}")
    return candidates[0]


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

    candidates = (
        [(short, "importer", "holo") for short in config.load_importers()]
        + [(short, "non_importer", "apo") for short in config.load_non_importers()]
    )

    for short, role, status in candidates:
        if role == "importer" and ligand_for(short) != config.LIGAND_KEY:
            skipped.append(f"{short}: assigned ligand is {ligand_for(short)!r} in this pipeline, "
                            f"not {config.LIGAND_KEY!r} -- its holoform conformation was never "
                            f"challenged with GA1, and there's no GA1 ABCfold pose to compare against")
            continue
        full = _full_protein_name(short)
        if full is None:
            skipped.append(f"{short}: no tm_reannotated clustering output found at all")
            continue
        if not config.has_cdd_residues(full):
            skipped.append(f"{full}: no CDD pocket residues (see config.load_cdd_residues docstring)")
            continue
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
    print(f"Wrote {len(rows)} rows to {config.MANIFEST_CSV} "
          f"({n_importer} importer, {n_non_importer} non_importer)")
    for r in rows:
        print(f"  {r['role']:>13s}  {r['protein']:<16s}  ca{r['ca_cluster']}  {r['receptor_cif']}")

    if skipped:
        print(f"\nSkipped {len(skipped)} candidate(s):")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
