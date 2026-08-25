"""
redocking/src/extract_receptor_pdb.py
========================================
Stage 3: protein-only receptor PDB per manifest row, for HADDOCK3's
`[topoaa]` module. Same gemmi chain-selection + PDB serialization approach
as rescoring/src/pose_prep.py's `_protein_only_pdb_text` (mmCIF -> PDB via
gemmi, not text-slicing) -- reused directly rather than reimplemented,
since it already handles every ABCfold backend's mmCIF dialect.

Unlike pose_prep.py this never touches the ligand -- for the importer
complex, ABCfold's own co-folded GA1 pose is deliberately discarded here:
HADDOCK3 is meant to dock the RDKit/GA3-templated ligand
(data/ga1_from_ga3.sdf, build_ga1_from_ga3.py) fresh via its own
rigid-body sampling, independent of where the ab initio backend placed it.
The ABCfold GA1 pose is only ever used later, downstream in
compare_to_abcfold.py, as the thing HADDOCK3's redocked pose gets compared
against -- not as a starting point.

Output: data/receptors/<complex_id>_receptor.pdb.
"""
from __future__ import annotations

from pathlib import Path

import gemmi

import config

RECEPTOR_DIR = config.DATA_DIR / "receptors"


def extract_receptor_pdb(cif_path: Path, out_path: Path, protein_chain: str = config.PROTEIN_CHAIN) -> Path:
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    selection = gemmi.Selection(f"/1/{protein_chain}")
    protein_structure = selection.copy_structure_selection(structure)
    if len(protein_structure) == 0 or len(protein_structure[0]) == 0:
        raise ValueError(f"{cif_path}: no atoms found in chain {protein_chain!r} (protein chain)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    protein_structure.write_pdb(str(out_path))
    return out_path


def main() -> None:
    import csv
    RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cif_path = config.PIPELINE_ROOT / row["receptor_cif"]
        out_path = RECEPTOR_DIR / f"{row['complex_id']}_receptor.pdb"
        extract_receptor_pdb(cif_path, out_path)
        print(f"{row['complex_id']}: {cif_path} -> {out_path}")


if __name__ == "__main__":
    main()
