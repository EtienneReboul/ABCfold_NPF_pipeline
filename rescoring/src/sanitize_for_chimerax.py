#!/usr/bin/env python3
"""
rescoring/src/sanitize_for_chimerax.py
=========================================
Prepares one complex's staged pose (protein + bond-order-corrected ligand,
see ligand_fix.py/pose_prep.py) for ChimeraX minimization (chimerax_minimize_pose.py),
with an explicit ligand-geometry sanity check first.

Why a separate check here, even though pose_prep.py already reuses the
SMILES-corrected chemistry -- strictly better than the sibling
NPF_pocket_pipeline project's sanitize_cif.py, which has to re-derive bond
orders from raw-distance proximity bonding + patch over-valent atoms after
the fact: correct bond ORDERS don't guarantee sane bond LENGTHS or contact
distances. A raw ABCfold pose can still place the ligand's own atoms
implausibly close/far apart (a bad co-folding hypothesis), and ChimeraX's
minimizer can fail outright or blow up on that starting geometry. This
script fails loudly (raises, does not write a staged PDB) on any bond
stretched/collapsed relative to covalent-radii sums, or any non-bonded
heavy-atom pair in steric clash, instead of silently handing a broken pose
to the minimizer -- same "fail loudly rather than pool a possibly-wrong
result" convention the rest of rescoring/ uses (see ligand_fix.py's own
element-sequence check, which this reuses via build_corrected_ligand_mol).

Usage:
    python sanitize_for_chimerax.py --complex-id <id> --out <staged.pdb>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

import config
import ligand_fix as lf
import pose_prep as pp

# Flag any bond longer than this multiple of (r_cov_i + r_cov_j) as stretched,
# or shorter than this multiple as collapsed onto itself.
BOND_STRETCH_FACTOR = 1.8
BOND_CLASH_FACTOR = 0.4
# Flag any non-bonded heavy-atom pair closer than this multiple of
# (r_cov_i + r_cov_j) as a steric clash within the ligand's own geometry.
NONBONDED_CLASH_FACTOR = 0.55


class LigandGeometryError(ValueError):
    pass


def check_ligand_geometry(mol: Chem.Mol) -> list[str]:
    """Heavy-atom bond-length and non-bonded-contact sanity check on one
    pose's corrected ligand mol (real predicted coordinates, correct bond
    orders from ligand_fix.py). Hydrogens are excluded -- Chem.AddHs's
    addCoords placement is a geometric heuristic, not a physical one, so
    routine pre-minimization H clashes are expected and always resolved by
    the minimizer itself; they are not a sign of a broken heavy-atom pose."""
    pt = Chem.GetPeriodicTable()
    conf = mol.GetConformer()
    issues = []

    bonded_pairs = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_pairs.add(frozenset((i, j)))
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        if ai.GetAtomicNum() == 1 or aj.GetAtomicNum() == 1:
            continue
        expected = pt.GetRcovalent(ai.GetAtomicNum()) + pt.GetRcovalent(aj.GetAtomicNum())
        actual = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
        if actual > BOND_STRETCH_FACTOR * expected:
            issues.append(f"bond {ai.GetSymbol()}{i}-{aj.GetSymbol()}{j} stretched: "
                           f"{actual:.2f} A (expected ~{expected:.2f} A)")
        elif actual < BOND_CLASH_FACTOR * expected:
            issues.append(f"bond {ai.GetSymbol()}{i}-{aj.GetSymbol()}{j} collapsed: "
                           f"{actual:.2f} A (expected ~{expected:.2f} A)")

    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    for pos, i in enumerate(heavy_idx):
        for j in heavy_idx[pos + 1:]:
            if frozenset((i, j)) in bonded_pairs:
                continue
            ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
            expected = pt.GetRcovalent(ai.GetAtomicNum()) + pt.GetRcovalent(aj.GetAtomicNum())
            actual = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
            if actual < NONBONDED_CLASH_FACTOR * expected:
                issues.append(f"non-bonded heavy-atom clash {ai.GetSymbol()}{i}-{aj.GetSymbol()}{j}: "
                               f"{actual:.2f} A")

    return issues


def resolve_complex(complex_id: str) -> dict:
    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows = manifest[manifest["complex_id"] == complex_id]
    if rows.empty:
        sys.exit(f"complex_id {complex_id!r} not found in {config.MANIFEST_CSV}")
    row = rows.iloc[0]
    ligand_chain, smiles = pp.resolved_ligand_chain(config.ABCFOLD_OUT_ROOT / f"{row['protein']}__holo")
    template = lf.build_template(smiles, config.ligand_resname(row["ligand"]))
    return {
        "cif_path": config.PIPELINE_ROOT / row["cif_path"],
        "ligand_chain": ligand_chain,
        "template": template,
        "protein": row["protein"],
        "ligand": row["ligand"],
    }


def sanitize_and_stage(complex_id: str, out_path: Path) -> Path:
    """Resolve complex_id -> corrected ligand mol -> geometry check ->
    staged PDB (raises LigandGeometryError instead of writing out_path if
    the check fails)."""
    info = resolve_complex(complex_id)
    mol = lf.build_corrected_ligand_mol(info["cif_path"], info["ligand_chain"], info["template"])
    issues = check_ligand_geometry(mol)
    if issues:
        raise LigandGeometryError(
            f"{complex_id}: ligand geometry looks broken, refusing to hand this pose to "
            f"ChimeraX's minimizer ({len(issues)} issue(s)):\n  " + "\n  ".join(issues)
        )
    return pp.prepare_complex_pdb(info["cif_path"], info["ligand_chain"], info["template"], out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex-id", required=True)
    ap.add_argument("--out", required=True, help="output staged PDB path")
    args = ap.parse_args()

    try:
        out_path = sanitize_and_stage(args.complex_id, Path(args.out))
    except LigandGeometryError as e:
        sys.exit(f"[sanitize_for_chimerax] {e}")
    print(f"[sanitize_for_chimerax] wrote {out_path}")


if __name__ == "__main__":
    main()
