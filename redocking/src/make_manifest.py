"""
redocking/src/make_manifest.py
=================================
Stage 2: this run's receptor complexes -- every NPF_LDA_kernel HC importer
that has a GA1-holoform cluster-representative pose in this pipeline
(positive controls) + every HC non-importer that has an apoform ABCfold
structure (negative controls), filtered down to only those that ALSO have
CDD pocket residues (config.has_cdd_residues -- see that function's
docstring: a handful of proteins have a genuinely empty InterProScan
result, confirmed by hand, not just a not-yet-run one).

Coverage is NOT 1:1 with NPF_LDA_kernel's full hc_importers/
hc_non_importers lists -- confirmed by hand (2026-08-25) against this
pipeline's actual data:
  - Only 5 of 12 hc_importers have a GA1-holoform pose in THIS pipeline
    (NPF3.1, NPF2.12, NPF2.13, NPF2.10, NPF2.5). The other 7 were either
    co-folded with a different assigned ligand here instead (NPF2.7/
    NPF2.3/NPF2.4/NPF1.1/NPF1.2 -> nitrate, NPF4.2 -> ABA -- see
    scripts/ligand_assignment.py's HC_IMPORTERS/NITRATE_TRANSPORTERS/
    ABA_TRANSPORTERS), or have a GA1 pose but no CDD residues (NPF4.1).
    Getting a GA1 pose for the first 6 needs NEW ABCfold cofolding runs
    (out of scope here); NPF4.1's CDD gap is a genuine empty InterProScan
    result, not a pending one (see config.load_cdd_residues's docstring).
  - 19 of 21 hc_non_importers are usable; NPF8.5 and NPF5.9 have the same
    genuine-empty-CDD-result issue as NPF4.1.

Importer side reuses rescoring/data/manifest.csv's already-computed
cluster assignments (rescoring/src/make_manifest.py owns that clustering
logic; this script only reads its output) -- deterministic pick: among a
protein's rows for GA1-holoform, sort by (ca_cluster, ligand_pose_cluster,
complex_id) and take the first.

Output: data/manifest.csv (complex_id, protein, role [importer/
non_importer], form [holo/apo], receptor_cif) + a coverage report printed
to stdout (skipped proteins + why).
"""
from __future__ import annotations

import csv
from pathlib import Path

import config


def _full_protein_name(short_name: str) -> str | None:
    """'NPF2.10' -> 'NPF2.10_Q944G5', by matching against whichever real
    protein directories/manifest rows exist -- avoids hardcoding UniProt
    accessions here. None if no match found anywhere."""
    candidates = set()
    if config.ABCFOLD_OUT_ROOT.exists():
        for d in config.ABCFOLD_OUT_ROOT.iterdir():
            name = d.name.rsplit("__", 1)[0]
            if name.startswith(short_name + "_"):
                candidates.add(name)
    if config.RESCORING_MANIFEST_CSV.exists():
        with config.RESCORING_MANIFEST_CSV.open() as f:
            for row in csv.DictReader(f):
                if row["protein"].startswith(short_name + "_"):
                    candidates.add(row["protein"])
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"{short_name!r} matched multiple full names: {sorted(candidates)}")
    return candidates.pop()


def _importer_cluster_rep(protein: str, ligand_key: str = config.LIGAND_KEY) -> tuple[str, str] | None:
    """(complex_id, cif_path) for `protein`'s lowest (ca_cluster,
    ligand_pose_cluster) `ligand_key`-holoform row in rescoring's own
    manifest.csv -- None if this protein has no such row (never co-folded
    with this ligand in this pipeline)."""
    with config.RESCORING_MANIFEST_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["protein"] == protein and r["ligand"] == ligand_key]
    if not rows:
        return None
    # ligand_pose_cluster is "" for rows whose ca_cluster never got a
    # ligand-pose sub-clustering pass (too few poses, or a different
    # macro-state) -- confirmed present in real data, not just a pilot
    # edge case. Sort those last (inf) rather than crashing on int("").
    rows.sort(key=lambda r: (int(r["ca_cluster"]),
                              int(r["ligand_pose_cluster"]) if r["ligand_pose_cluster"] else float("inf"),
                              r["complex_id"]))
    best = rows[0]
    return best["complex_id"], best["cif_path"]


def build_manifest() -> tuple[list[dict], list[str]]:
    if not config.RESCORING_MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{config.RESCORING_MANIFEST_CSV} not found -- run rescoring/src/make_manifest.py first "
            f"(this reuses its cluster-representative selection, not its own copy of the logic)."
        )

    manifest_rows: list[dict] = []
    skipped: list[str] = []

    for short in config.load_importers():
        full = _full_protein_name(short)
        if full is None:
            skipped.append(f"{short}: no protein directory/manifest entry found at all")
            continue
        rep = _importer_cluster_rep(full)
        if rep is None:
            skipped.append(f"{full}: no GA1-holoform pose in rescoring/data/manifest.csv "
                            f"(co-folded with a different ligand in this pipeline, or never run)")
            continue
        if not config.has_cdd_residues(full):
            skipped.append(f"{full}: no CDD pocket residues (see config.load_cdd_residues docstring)")
            continue
        complex_id, cif_path = rep
        manifest_rows.append({
            "complex_id": complex_id, "protein": full, "role": "importer",
            "form": "holo", "receptor_cif": cif_path,
        })

    for short in config.load_non_importers():
        full = _full_protein_name(short)
        if full is None:
            skipped.append(f"{short}: no protein directory/manifest entry found at all")
            continue
        apo_dir = config.receptor_holo_apo_dir(full, "apo")
        if not apo_dir.exists():
            skipped.append(f"{full}: no apoform ABCfold run ({apo_dir})")
            continue
        if not config.has_cdd_residues(full):
            skipped.append(f"{full}: no CDD pocket residues (see config.load_cdd_residues docstring)")
            continue
        cif_candidates = sorted(apo_dir.glob("*/seed-*/*_model.cif"))
        if not cif_candidates:
            skipped.append(f"{full}: apoform dir exists but no *_model.cif found under it")
            continue
        cif_path = cif_candidates[0]
        manifest_rows.append({
            "complex_id": f"{full}__apo_{cif_path.parent.name}", "protein": full,
            "role": "non_importer", "form": "apo",
            "receptor_cif": str(cif_path.relative_to(config.PIPELINE_ROOT)),
        })

    return manifest_rows, skipped


def main() -> None:
    rows, skipped = build_manifest()
    with config.MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["complex_id", "protein", "role", "form", "receptor_cif"])
        writer.writeheader()
        writer.writerows(rows)

    n_importer = sum(1 for r in rows if r["role"] == "importer")
    n_non_importer = sum(1 for r in rows if r["role"] == "non_importer")
    print(f"Wrote {len(rows)} rows to {config.MANIFEST_CSV} "
          f"({n_importer} importer, {n_non_importer} non_importer)")
    for r in rows:
        print(f"  {r['role']:>13s}  {r['protein']:<16s}  {r['receptor_cif']}")

    if skipped:
        print(f"\nSkipped {len(skipped)} candidate(s):")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
