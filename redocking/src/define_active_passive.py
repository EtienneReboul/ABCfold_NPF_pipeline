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
structure. `haddock3-restraints active_passive_to_ambig` then turns two
per-molecule "actpass" files into the .tbl file `[rigidbody]`/`[flexref]`
consume.

**CLI contract, confirmed by hand against a real install on the IFB
cluster (2026-08-25)** -- not guessable from --help alone, so recorded
here:
  - `passive_from_active <structure.pdb> <active_list>`: `active_list` is
    a LITERAL comma-separated string of residue numbers passed directly
    as a CLI argument (e.g. "78,79,82"), NOT a file path. Output: passive
    residue numbers, space-separated, on stdout.
  - `active_passive_to_ambig <actpass_one> <actpass_two> [--segid-one A]
    [--segid-two B]`: each `actpass` argument is a path to ONE combined
    file per molecule, EXACTLY two lines -- line 1 = active residues
    (space-separated ints), line 2 = passive residues (space-separated
    ints, can be empty) -- `haddock.libs.librestraints.parse_actpass_file`
    raises if a file doesn't have exactly 2 lines. Default segids ("A"/"B")
    match make_haddock_cfg.py's molecule order (receptor first, ligand
    second) and compare_to_abcfold.py's `LIGAND_CHAIN_HADDOCK = "B"`
    assumption -- confirmed consistent, not just guessed twice the same way.
    Output: the ambig.tbl text, on stdout.

The ligand side (GA1, a single-molecule "chain") is given active=[1]
(its one residue), passive=[] -- the whole ligand is the interacting
partner, there's no separate passive shell for a 25-heavy-atom small
molecule.

Requires HADDOCK3 installed (for the `haddock3-restraints` CLI) -- this
script shells out to it rather than reimplementing its residue-selection
logic, so it needs the redocking conda env (envs/redocking.yaml) active.

Output per complex: data/restraints/<complex_id>_receptor_actpass.txt,
<complex_id>_ligand_actpass.txt, <complex_id>_ambig.tbl.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import config
from extract_receptor_pdb import RECEPTOR_DIR

RESTRAINTS_DIR = config.DATA_DIR / "restraints"
LIGAND_RESIDUE_NUM = 1  # GA1 is a single-residue "chain" in the ligand PDB
SEGID_RECEPTOR = "A"
SEGID_LIGAND = "B"


def _require_haddock3_restraints() -> str:
    exe = shutil.which("haddock3-restraints")
    if exe is None:
        raise RuntimeError(
            "haddock3-restraints not found on PATH -- activate the redocking conda env "
            "(envs/redocking.yaml) first. This script deliberately does not reimplement "
            "its passive_from_active/active_passive_to_ambig logic."
        )
    return exe


def _write_actpass_file(path: Path, active: list[int], passive: list[int]) -> Path:
    """Exactly two lines -- see module docstring's CLI contract notes."""
    path.write_text(" ".join(str(r) for r in active) + "\n" + " ".join(str(r) for r in passive) + "\n")
    return path


def define_active_passive(protein: str, receptor_pdb: Path, out_stub: Path) -> tuple[Path, Path, Path]:
    haddock3_restraints = _require_haddock3_restraints()

    active_residues = config.load_cdd_residues(protein)
    active_csv = ",".join(str(r) for r in active_residues)

    passive_result = subprocess.run(
        [haddock3_restraints, "passive_from_active", str(receptor_pdb), active_csv],
        capture_output=True, text=True, check=True,
    )
    passive_residues = [int(x) for x in passive_result.stdout.split()]

    receptor_actpass = _write_actpass_file(
        out_stub.with_name(out_stub.name + "_receptor_actpass.txt"), active_residues, passive_residues)
    ligand_actpass = _write_actpass_file(
        out_stub.with_name(out_stub.name + "_ligand_actpass.txt"), [LIGAND_RESIDUE_NUM], [])

    ambig_path = out_stub.with_name(out_stub.name + "_ambig.tbl")
    ambig_result = subprocess.run(
        [haddock3_restraints, "active_passive_to_ambig", str(receptor_actpass), str(ligand_actpass),
         "--segid-one", SEGID_RECEPTOR, "--segid-two", SEGID_LIGAND],
        capture_output=True, text=True, check=True,
    )
    ambig_path.write_text(ambig_result.stdout)

    return receptor_actpass, ligand_actpass, ambig_path


def main() -> None:
    RESTRAINTS_DIR.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        receptor_pdb = RECEPTOR_DIR / f"{row['complex_id']}_receptor.pdb"
        out_stub = RESTRAINTS_DIR / row["complex_id"]
        receptor_actpass, ligand_actpass, ambig_path = define_active_passive(row["protein"], receptor_pdb, out_stub)
        print(f"{row['complex_id']}: receptor={receptor_actpass.name} ligand={ligand_actpass.name} "
              f"ambig={ambig_path.name}")


if __name__ == "__main__":
    main()
