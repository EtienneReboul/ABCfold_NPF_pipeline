"""
redocking/src/define_active_passive.py
=========================================
Stage 4: HADDOCK3 ambiguous interaction restraints (AIRs) per manifest
complex.

"Active" residues = the CDD/InterPro putative pocket residues
NPF_pocket_pipeline already computed (config.load_cdd_residues -- read
directly from that project's data/interpro/cdd_summary.json, never
recomputed here). "Passive" residues are NOT hand-derived with a
distance-cutoff script -- HADDOCK3 ships a CLI tool for exactly this,
`haddock3-restraints passive_from_active`, which surface-restricts and
neighbor-expands an active-residue list against the actual receptor
structure. Using it directly means the passive selection follows the same
convention every other HADDOCK3 protocol uses, instead of inventing a new
one. `haddock3-restraints active_passive_to_ambig` then turns both lists
into the .tbl file `[rigidbody]`/`[flexref]` consume.

Requires HADDOCK3 installed (for the `haddock3-restraints` CLI) -- this
script shells out to it rather than reimplementing its residue-selection
logic, so it needs the redocking conda env (envs/redocking.yaml) active.

Output per complex: data/restraints/<complex_id>_active.txt,
<complex_id>_passive.txt, <complex_id>_ambig.tbl.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import config
from extract_receptor_pdb import RECEPTOR_DIR

RESTRAINTS_DIR = config.DATA_DIR / "restraints"


def _require_haddock3_restraints() -> str:
    exe = shutil.which("haddock3-restraints")
    if exe is None:
        raise RuntimeError(
            "haddock3-restraints not found on PATH -- activate the redocking conda env "
            "(envs/redocking.yaml) first. This script deliberately does not reimplement "
            "its passive_from_active/active_passive_to_ambig logic."
        )
    return exe


def define_active_passive(protein: str, receptor_pdb: Path, out_stub: Path) -> tuple[Path, Path, Path]:
    haddock3_restraints = _require_haddock3_restraints()

    active_residues = config.load_cdd_residues(protein)
    active_path = out_stub.with_name(out_stub.name + "_active.txt")
    active_path.write_text(" ".join(str(r) for r in active_residues) + "\n")

    passive_path = out_stub.with_name(out_stub.name + "_passive.txt")
    passive_result = subprocess.run(
        [haddock3_restraints, "passive_from_active", str(receptor_pdb), str(active_path)],
        capture_output=True, text=True, check=True,
    )
    passive_path.write_text(passive_result.stdout)

    ambig_path = out_stub.with_name(out_stub.name + "_ambig.tbl")
    ambig_result = subprocess.run(
        [haddock3_restraints, "active_passive_to_ambig", str(active_path), str(passive_path)],
        capture_output=True, text=True, check=True,
    )
    ambig_path.write_text(ambig_result.stdout)

    return active_path, passive_path, ambig_path


def main() -> None:
    RESTRAINTS_DIR.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        receptor_pdb = RECEPTOR_DIR / f"{row['complex_id']}_receptor.pdb"
        out_stub = RESTRAINTS_DIR / row["complex_id"]
        active_path, passive_path, ambig_path = define_active_passive(row["protein"], receptor_pdb, out_stub)
        print(f"{row['complex_id']}: active={active_path.name} passive={passive_path.name} ambig={ambig_path.name}")


if __name__ == "__main__":
    main()
