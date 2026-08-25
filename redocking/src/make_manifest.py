"""
redocking/src/make_manifest.py
=================================
Stage 2 (pilot): pick this pilot's receptor complexes -- one confirmed GA1
importer (HC_IMPORTERS, cluster-representative holoform pose) + the two
known non-importers already confirmed to have apoform ABCfold structures
and CDD pocket annotations (NPF6.1, NPF8.1).

Reuses rescoring/data/manifest.csv's already-computed cluster assignments
for the importer side (rescoring/src/make_manifest.py owns that clustering
logic; this script only reads its output) rather than re-deriving a
cluster-representative pose. Deterministic pick: among rows for the chosen
importer + GA1, sort by (ca_cluster, ligand_pose_cluster, complex_id) and
take the first -- the macro-state/pose-cluster 0 representative, same
"lowest cluster id" convention that cluster_conformations.py's own
representative-selection already establishes.

Non-importer receptors have no ligand-pose clustering (they're apoform --
no ligand was ever co-folded), so their manifest rows just point straight
at results/abcfold/<protein>__apo/.

Output: data/manifest.csv (complex_id, protein, role [importer/non_importer],
receptor_cif, form [holo/apo]).
"""
from __future__ import annotations

import csv

import config

# Pilot scope -- see redocking plan: 1 confirmed HC importer (positive
# control) + 2 confirmed non-importers (negative control), both sides
# already have everything this pipeline needs (holoform cluster-rep pose /
# apoform structure, CDD pocket residues -- see config.load_cdd_residues).
PILOT_IMPORTER = "NPF2.10_Q944G5"
PILOT_NON_IMPORTERS = ["NPF6.1_Q9LYR6", "NPF8.1_Q9M390"]


def _importer_cluster_rep(protein: str, ligand_key: str = config.LIGAND_KEY) -> tuple[str, str]:
    """(complex_id, cif_path) for `protein`'s lowest (ca_cluster,
    ligand_pose_cluster) `ligand_key`-holoform row in rescoring's own
    manifest.csv."""
    if not config.RESCORING_MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{config.RESCORING_MANIFEST_CSV} not found -- run rescoring/src/make_manifest.py first "
            f"(this pilot reuses its cluster-representative selection, not its own copy of the logic)."
        )
    with config.RESCORING_MANIFEST_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["protein"] == protein and r["ligand"] == ligand_key]
    if not rows:
        raise ValueError(f"No {ligand_key} rows for {protein!r} in {config.RESCORING_MANIFEST_CSV}")
    rows.sort(key=lambda r: (int(r["ca_cluster"]), int(r["ligand_pose_cluster"]), r["complex_id"]))
    best = rows[0]
    return best["complex_id"], best["cif_path"]


def build_manifest() -> list[dict]:
    manifest_rows = []

    complex_id, cif_path = _importer_cluster_rep(PILOT_IMPORTER)
    manifest_rows.append({
        "complex_id": complex_id,
        "protein": PILOT_IMPORTER,
        "role": "importer",
        "form": "holo",
        "receptor_cif": cif_path,
    })

    for protein in PILOT_NON_IMPORTERS:
        apo_dir = config.receptor_holo_apo_dir(protein, "apo")
        if not apo_dir.exists():
            raise FileNotFoundError(f"{apo_dir} not found -- expected an existing apoform ABCfold run for "
                                     f"{protein!r} (this pilot only covers non-importers already present in "
                                     f"results/abcfold/).")
        # Any single apo model is fine here -- unlike the importer side there
        # is no ligand pose to cluster on; pick the first seed/sample
        # deterministically (sorted glob) as the receptor conformation.
        cif_candidates = sorted(apo_dir.glob("*/seed-*/*_model.cif"))
        if not cif_candidates:
            raise FileNotFoundError(f"No *_model.cif found under {apo_dir}")
        cif_path = cif_candidates[0]
        manifest_rows.append({
            "complex_id": f"{protein}__apo_{cif_path.parent.name}",
            "protein": protein,
            "role": "non_importer",
            "form": "apo",
            "receptor_cif": str(cif_path.relative_to(config.PIPELINE_ROOT)),
        })

    return manifest_rows


def main() -> None:
    rows = build_manifest()
    with config.MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["complex_id", "protein", "role", "form", "receptor_cif"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {config.MANIFEST_CSV}")
    for r in rows:
        print(f"  {r['role']:>13s}  {r['protein']:<16s}  {r['receptor_cif']}")


if __name__ == "__main__":
    main()
