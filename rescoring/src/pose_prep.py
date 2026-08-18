"""
rescoring/src/pose_prep.py
=============================
Combines one complex's protein chain (as-is, straight from its ABCfold CIF)
with the bond-order-corrected ligand (see ligand_fix.py) into one staged
PDB file ready for PyRosetta (`-extra_res_fa params/<ligand>.params`).

Generalized from the sibling project's PDB-text-slicing version: the source
here is an ABCfold mmCIF (any of 6 backends), so protein atoms are extracted
via `gemmi` (chain selection + PDB serialization) instead of filtering an
already-PDB-formatted pose. The ligand chain id is resolved per protein from
its own `abc_fold_input.resolved.json` (same lookup
`scripts/_notebook_setup_functions.py`'s `_resolved_ligand_info` and
`scripts/cluster_conformations.py`'s copy of it already use) rather than
assumed fixed, since it's read from ABCfold's own resolved fold-input record
rather than hardcoded.

Output staged-PDB convention (residue "LIG", chain "L", protein chain "A")
is unchanged from the sibling project, so relief.py/decompose.py need zero
changes.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gemmi
from rdkit import Chem

import config
import ligand_fix as lf


def resolved_ligand_chain(protein_holo_run_dir: Path) -> tuple[str, str]:
    """(ligand_chain_id, smiles) for one holoform run, read from ABCfold's
    own resolved fold-input JSON -- the authoritative, backend-agnostic
    record of which chain id ABCfold assigned the ligand."""
    path = protein_holo_run_dir / "abc_fold_input.resolved.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- no resolved holoform fold-input for this run")
    data = json.loads(path.read_text())
    for seq in data["sequences"]:
        if "ligand" in seq:
            return seq["ligand"]["id"][0], seq["ligand"]["smiles"]
    raise ValueError(f"{path} has no 'ligand' entry in its sequences list")


def load_template(smiles: str, resname: str) -> Chem.Mol:
    return lf.build_template(smiles, resname)


def _protein_only_pdb_text(cif_path: Path, protein_chain: str = config.PROTEIN_CHAIN) -> str:
    """Select just `protein_chain` out of `cif_path` and serialize it to PDB
    text via gemmi (handles mmCIF -> PDB atom/residue formatting directly,
    so no per-backend column-format handling is needed here)."""
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    selection = gemmi.Selection(f"/1/{protein_chain}")
    protein_structure = selection.copy_structure_selection(structure)
    if len(protein_structure) == 0 or len(protein_structure[0]) == 0:
        raise ValueError(f"{cif_path}: no atoms found in chain {protein_chain!r} (protein chain)")

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as tmp:
        protein_structure.write_pdb(tmp.name)
        return Path(tmp.name).read_text()


def _protein_lines(pdb_text: str) -> list[str]:
    """ATOM/TER lines only -- CONECT/END dropped (ligand CONECT is
    regenerated fresh below; the protein-only selection shouldn't carry any
    of its own, but this stays defensive)."""
    out = []
    for l in pdb_text.splitlines():
        if l.startswith("CONECT") or l.startswith("END") or l.startswith(("HEADER", "CRYST1", "REMARK")):
            continue
        if l.startswith(("ATOM", "TER")):
            out.append(l)
    return out


def _renumber_ligand_block(pdb_block: str, start_serial: int) -> list[str]:
    """Shift HETATM/CONECT serials in an RDKit-written ligand PDB block past start_serial."""
    lines = [l for l in pdb_block.splitlines() if l.strip() and not l.startswith(("END", "MASTER"))]
    old_to_new: dict[int, int] = {}
    next_serial = start_serial
    out = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            old = int(l[6:11])
            old_to_new[old] = next_serial
            new_line = f"HETATM{next_serial:>5d}{l[11:]}"
            out.append(new_line)
            next_serial += 1
    for l in lines:
        if l.startswith("CONECT"):
            fields = [int(x) for x in l[6:].split()]
            new_fields = [old_to_new[fields[0]]] + [old_to_new[f] for f in fields[1:]]
            out.append("CONECT" + "".join(f"{f:>5d}" for f in new_fields))
    return out


def prepare_complex_pdb(cif_path: Path, ligand_chain_id: str, template: Chem.Mol, out_path: Path) -> Path:
    """Write a staged PDB (protein as-is + bond-order-corrected, Rosetta-named
    ligand) to out_path, sourced directly from one ABCfold CIF."""
    protein_text = _protein_only_pdb_text(cif_path)
    protein_lines = _protein_lines(protein_text)

    max_serial = 0
    for l in protein_lines:
        if l.startswith("ATOM"):
            max_serial = max(max_serial, int(l[6:11]))

    ligand_block = lf.corrected_ligand_pdb_block(cif_path, ligand_chain_id, template)
    ligand_lines = _renumber_ligand_block(ligand_block, max_serial + 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(protein_lines + ligand_lines + ["END", ""]))
    return out_path
