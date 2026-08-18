"""
rescoring/src/ligand_fix.py
==============================
Builds a chemically-correct ligand RDKit mol (correct bond orders + explicit
hydrogens) for one ABCfold-predicted pose, from the ligand's canonical
SMILES (config.yaml's `ligands:` dict) plus that pose's own 3-D heavy-atom
coordinates.

Why this exists: none of the 6 ABCfold backends' raw CIF ligand output
carries bond-order records (this pipeline's whole downstream analysis
already works around that the same way the sibling NPF_pocket_pipeline
project's `sanitize_cif.py`/PLIP tooling does — pure-distance RDKit bond
perception, which produces zero double bonds for anything with an sp2
center). PyRosetta needs correct bond orders + hydrogens to score a ligand
at all, so this module reconstructs them per pose from the known-correct
SMILES template, the same "local, rescoring-only fix" scoping the sibling
project used (never touches results/abcfold/ itself).

Generalized from the sibling project's GA1-only, single-backend (Boltz-2)
version, which matched heavy atoms **by PDB atom name** against one
reference pose's own CONECT-derived bond list. That doesn't hold here:
spot-checked across all 6 backends for a real GA1 complex
(NPF2.12_Q9LFX9__holo), every backend uses its OWN, mutually inconsistent
ligand atom-naming/numbering convention (AlphaFold3 `LIG_B` with names like
`C1..C19`, Boltz `LIG1` with `C38..C49`-style global numbering, chai1
`LIG2` with `_1`-suffixed names, protenix `l01`, rosettafold `L:0` with
0-indexed names) AND a different `label_comp_id` (residue name) per
backend — so nothing about a fixed "LIG" resname or fixed atom names
generalizes across backends.

What IS backend-invariant (same spot-check): every backend places the
ligand's heavy atoms in the exact same **positional** (file) order, by
element — identical across all 6 backends and identical to
`Chem.MolFromSmiles(smiles)`'s own canonical atom order. So instead of
matching atoms by name against a reference-derived bond list, this module
matches purely by POSITION: build each pose's corrected mol by taking the
SMILES template's own (already-correct) bonds/atoms and just attaching that
one pose's 3-D coordinates positionally, after verifying the pose's
heavy-atom element sequence matches the template's exactly (fails loudly,
per-pose, if it doesn't — e.g. a ligand this positional-order assumption
turns out not to hold for). This also means the sibling project's
reference-pose-derived `heavy_bonds_by_name` bookkeeping is unnecessary
here: the template mol already has correct bonds from the SMILES, nothing
needs to be re-derived or cached per ligand beyond the SMILES itself.

Atom naming: heavy atoms get one canonical name per ligand (an
element+running-index scheme, e.g. "C1".."C19", "O1".."O6" for GA1),
assigned once in the SMILES's own atom order and reused unchanged by every
pose/backend of that ligand — independent of whatever name any given
backend happened to use. Hydrogens `Chem.AddHs` adds are named `H01..H<n>`
in RDKit's deterministic traversal order (same convention the sibling
project used), which stays fixed across poses because the heavy-atom order
+ bond topology are fixed (from the template, not per-pose perception).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import config

LIG_CHAIN = config.LIGAND_CHAIN


@dataclass
class HeavyAtom:
    element: str
    x: float
    y: float
    z: float


def build_template(smiles: str, resname: str) -> Chem.Mol:
    """Heavy-atom-only mol from the canonical SMILES (correct bond orders,
    aromaticity, no Hs) with canonical PDB atom names assigned once, in the
    SMILES's own atom order -- see module docstring. `resname` is this
    ligand's own distinct Rosetta residue code (config.ligand_resname) --
    every downstream function derives it back from the template's own atoms
    rather than taking it as a separate parameter, see _resname_of."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"RDKit could not parse ligand SMILES: {smiles}")

    counts: dict[str, int] = {}
    for atom in template.GetAtoms():
        elem = atom.GetSymbol()
        counts[elem] = counts.get(elem, 0) + 1
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{elem}{counts[elem]}".ljust(4))
        info.SetResidueName(resname)
        info.SetChainId(LIG_CHAIN)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    return template


def _resname_of(template: Chem.Mol) -> str:
    return template.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip()


def parse_ligand_heavy_atoms(cif_path: Path, ligand_chain_id: str) -> list[HeavyAtom]:
    """Ligand heavy atoms (element, xyz), in FILE order -- see module
    docstring for why file/positional order (not atom name) is what's
    trusted to correspond to the template's own atom order."""
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    if ligand_chain_id not in {c.name for c in model}:
        raise ValueError(f"{cif_path}: no chain {ligand_chain_id!r} (ligand chain) in this CIF")
    atoms = []
    for residue in model[ligand_chain_id]:
        for atom in residue:
            if atom.element.name == "H":
                continue
            atoms.append(HeavyAtom(atom.element.name, atom.pos.x, atom.pos.y, atom.pos.z))
    return atoms


def build_corrected_ligand_mol(cif_path: Path, ligand_chain_id: str, template: Chem.Mol) -> Chem.Mol:
    """Full per-pose correction: one pose's raw CIF ligand chain -> corrected
    RDKit mol (template's bonds/names + this pose's own 3-D coordinates,
    matched positionally) -> Chem.AddHs. Raises ValueError if this pose's
    heavy-atom element sequence doesn't match the template's -- i.e. the
    positional-correspondence assumption this whole module rests on doesn't
    hold for this specific pose (caller should skip/log, not silently pool
    a possibly-wrong correction into scoring)."""
    heavy_atoms = parse_ligand_heavy_atoms(cif_path, ligand_chain_id)
    template_elements = [a.GetSymbol() for a in template.GetAtoms()]
    pose_elements = [a.element for a in heavy_atoms]
    if pose_elements != template_elements:
        raise ValueError(
            f"{cif_path}: ligand heavy-atom element sequence doesn't match "
            f"the template's (positional correspondence assumption violated) "
            f"-- pose has {len(pose_elements)} atoms {pose_elements}, "
            f"template has {len(template_elements)} atoms {template_elements}"
        )

    rw = Chem.RWMol(template)
    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, a in enumerate(heavy_atoms):
        conf.SetAtomPosition(i, Point3D(a.x, a.y, a.z))
    rw.RemoveAllConformers()
    rw.AddConformer(conf, assignId=True)

    mol = Chem.AddHs(rw.GetMol(), addCoords=True)
    return _name_hydrogens(mol, n_heavy=template.GetNumAtoms(), resname=_resname_of(template))


def _name_hydrogens(mol: Chem.Mol, n_heavy: int, resname: str) -> Chem.Mol:
    """Name every atom from index n_heavy onward H01, H02, ... in traversal
    order -- deterministic given a fixed heavy-atom order + bond topology
    (both guaranteed identical across poses of the same ligand -- see
    module docstring), so this produces the exact same name for "the same"
    hydrogen in every pose, without needing to look anything up."""
    for i, atom in enumerate(mol.GetAtoms()):
        if i < n_heavy:
            continue
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"H{i - n_heavy + 1:02d}".ljust(4))
        info.SetResidueName(resname)
        info.SetChainId(LIG_CHAIN)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    return mol


def corrected_ligand_pdb_block(cif_path: Path, ligand_chain_id: str, template: Chem.Mol) -> str:
    """Same as build_corrected_ligand_mol, serialized to a PDB block (HETATM + CONECT)."""
    mol = build_corrected_ligand_mol(cif_path, ligand_chain_id, template)
    return Chem.MolToPDBBlock(mol)


def build_idealized_mol(template: Chem.Mol, seed: int = 42) -> Chem.Mol:
    """Chemistry-ideal 3-D geometry for this ligand (ETKDGv3 embed + MMFF94
    optimization, falling back to UFF if MMFF has no parameters for this
    ligand), used ONLY as the geometry template `prep_ligand.py` hands to
    rdkit_to_params -- never for scoring a real predicted pose (that always
    uses build_corrected_ligand_mol's own positionally-matched coordinates).

    Why: rdkit_to_params derives each atom's Rosetta ICOOR_INTERNAL entry
    (the internal bond length/angle/dihedral tree Rosetta uses to place
    that atom, including at scoring time to check/rebuild it) from whatever
    3-D geometry the mol it's given happens to have. One real ABCfold-
    predicted pose's coordinates are a co-folding *hypothesis*, not a
    physically refined structure -- fine for a large, floppy, low-symmetry
    molecule (confirmed on GA1/ABA/JA-Ile: real-pose-geometry params
    generation validated cleanly across every backend), but small,
    high-symmetry ligands amplify small geometric imperfections into a
    numerically unstable ICOOR tree: nitrate (NO3-, a 4-atom, near-planar,
    resonance-symmetric ion) generated from a real predicted pose reliably
    reproduced Rosetta's own `fill_missing_atoms` internal-coordinate-
    rebuild failure when loading ANY pose, including poses other than the
    one the params file was built from. Rebuilding from an idealized,
    RDKit-optimized conformer instead resolved it."""
    mol = Chem.AddHs(Chem.Mol(template))
    embed_params = AllChem.ETKDGv3()
    embed_params.randomSeed = seed
    embed_params.numThreads = 1  # ETKDG can use multiple threads by default, which reintroduces
                                  # nondeterminism even with a fixed randomSeed -- pin to 1 so a given
                                  # seed actually reproduces the same conformer (still not perfectly
                                  # deterministic across RDKit builds/versions, see
                                  # prep_ligand.py's retry-on-failure loop, which is the real safety net)
    if AllChem.EmbedMolecule(mol, embed_params) != 0:
        raise RuntimeError(f"RDKit could not embed a 3-D conformer for {Chem.MolToSmiles(template)!r}")
    try:
        if AllChem.MMFFOptimizeMolecule(mol) != 0:
            raise RuntimeError("MMFF did not converge")
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)
    return _name_hydrogens(mol, n_heavy=template.GetNumAtoms(), resname=_resname_of(template))
