#!/usr/bin/env python3
"""
rescoring/src/_prep_ligand_probe_subprocess.py
==================================================
Internal helper for prep_ligand.py's `_build_params_with_self_check` --
NOT meant to be run directly. Builds ONE idealized-conformer params attempt
(written to a per-attempt temp path, not the canonical params/<ligand>.params
location -- the parent process decides which attempt to keep) and probes it
by staging + loading + scoring every given real complex, all in a fresh,
isolated Python process.

Why a subprocess: PyRosetta registers each `-extra_res_fa` residue type by
NAME into a process-wide ResidueTypeSet, and only the FIRST load of a given
name in a process actually takes effect -- calling `pyrosetta.init()` again
with the SAME residue name, even pointing at a params FILE whose content has
since changed (e.g. a retry with a fresh idealized conformer), silently
re-uses whatever was registered the first time. So a retry-with-a-fresh-seed
loop running in-process against the same resname would just keep re-testing
the FIRST attempt's geometry forever. Each attempt gets its own subprocess
instead, so each one is that residue name's genuinely first (and only) load.

Reads a JSON args file (sys.argv[1]): {ligand_key, seed, ligand_chain,
out_params_path, cif_paths: [...]}. Writes a JSON result file (sys.argv[2]):
{n_atoms, formula, results: [[complex_id, ok, score_or_error], ...]}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

import config  # noqa: E402
import ligand_fix as lf  # noqa: E402
import pose_prep as pp  # noqa: E402
import prep_ligand as pl  # noqa: E402


def main():
    args = json.loads(Path(sys.argv[1]).read_text())
    ligand_key = args["ligand_key"]
    seed = args["seed"]
    ligand_chain = args["ligand_chain"]
    out_params_path = Path(args["out_params_path"])
    cif_paths = [Path(p) for p in args["cif_paths"]]

    resname = config.ligand_resname(ligand_key)
    template = lf.build_template(config.load_ligand_smiles(ligand_key), resname)
    ideal_mol = lf.build_idealized_mol(template, seed=seed)
    pl.generate_params(ideal_mol, ligand_key, resname, out_path=out_params_path)

    import pyrosetta
    pyrosetta.init(f"-extra_res_fa {out_params_path} -mute all")

    results = []
    for cif_path in cif_paths:
        staged = config.STAGED_DIR / f"_prep_probe_{cif_path.stem}.pdb"
        try:
            # ligand_fix.build_corrected_ligand_mol (called inside prepare_complex_pdb)
            # raises ValueError/AtomValenceException specifically when this pose's
            # heavy-atom element sequence doesn't match the template's -- i.e. the
            # cross-backend positional-order assumption this whole pipeline rests on
            # (see ligand_fix.py's module docstring). That's a real correctness bug,
            # categorically different from PyRosetta rejecting one real structure's
            # own geometry (ring pucker, fill_missing_atoms, ...) -- tagged "mismatch"
            # vs "pose" so the parent process can treat them differently (any single
            # mismatch is fatal; pose failures are tolerated as real-structure noise
            # up to 50% -- see validate_one_ligand).
            try:
                pp.prepare_complex_pdb(cif_path, ligand_chain, template, staged)
            except (ValueError, Chem.AtomValenceException) as e:
                results.append([cif_path.stem, False, f"mismatch: {e}"])
                continue
            pose = pyrosetta.pose_from_pdb(str(staged))
            score = pyrosetta.get_score_function()(pose)
            results.append([cif_path.stem, True, float(score)])
        except Exception as e:
            results.append([cif_path.stem, False, f"pose: {e}"])
        finally:
            staged.unlink(missing_ok=True)

    Path(sys.argv[2]).write_text(json.dumps({
        "n_atoms": ideal_mol.GetNumAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(ideal_mol),
        "results": results,
    }))


if __name__ == "__main__":
    main()
