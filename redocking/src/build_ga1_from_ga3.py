"""
redocking/src/build_ga1_from_ga3.py
=====================================
Stage 0: build a physically-grounded 3D GA1 (Gibberellin A1) conformer,
templated on GA3's (Gibberellin A3) real deposited crystal geometry rather
than a bare ETKDG embed -- GA1 and GA3 differ only in ring-A saturation
(GA3 has a C1=C2 ene-diol-lactone-like ring A; GA1's ring A is saturated
with a 3-beta-OH), so most of the molecule (the bicyclic lactone bridge,
the exocyclic-methylene D-ring, the carboxylic acid) is structurally
identical and its real, physically-relaxed geometry is worth reusing rather
than re-idealizing from scratch.

Source of GA3's real geometry: the RCSB Chemical Component Dictionary (CCD)
entry for GA3 (ligand id "GA3") carries BOTH the correct bond table
(explicit bond orders, not distance-perceived) and "model" atom coordinates
-- i.e. real coordinates from an actual deposited structure the CCD entry
cites (`pdbx_model_coordinates_db_code`, currently 3ED1 for GA3; NOT the
"ideal" Corina-generated conformer, which is is a synthetic/idealized
geometry with no ring pucker information). This is more direct and more
reliable than pulling a full PDB entry (e.g. 2ZSH) and re-perceiving bonds
from distances -- the CCD's own bond table is authoritative and atom names
already match 1:1 across every PDB entry containing GA3.

Method:
  1. Download GA3's CCD .cif -> RDKit mol with real coordinates + explicit
     bond orders (build_ga3_real_mol).
  2. Build a plain GA1 mol from config.yaml's GA1 SMILES (no 3D coords).
  3. Find the maximum common substructure (MCS) between them (rdFMCS) --
     expected to cover the whole molecule except ring A's C1/C2/C3
     saturation-vs-unsaturation difference.
  4. RDKit's AllChem.ConstrainedEmbed: embed GA1 with the MCS-matched atoms
     tethered to GA3's real coordinates, then MMFF-relax the rest (ring-A
     substitution + hydrogens) freely.
  5. Sanity-check the result (bond lengths vs. covalent-radii sums, no
     non-bonded heavy-atom clashes) before writing it out -- same category
     of check as rescoring/src/sanitize_for_chimerax.py's ligand-geometry
     guard, for the same reason: a bad embed here would silently corrupt
     every downstream docking run.

Output: data/ga1_from_ga3.sdf (single conformer) + data/ga1_from_ga3.log
(MCS coverage, tether RMSD, sanity-check results -- provenance record, same
role as rescoring/params/<ligand>_atom_naming.json).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

import config

CCD_DOWNLOAD_URL = "https://files.rcsb.org/ligands/download/{ccd_id}.cif"
CACHE_DIR = config.DATA_DIR / "_cache"

_BOND_ORDER = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}

# Max acceptable stretch of a bond beyond the sum of its atoms' covalent
# radii, and the minimum allowed non-bonded heavy-atom distance -- same
# tolerances in spirit as sanitize_for_chimerax.py's ligand-geometry guard,
# tuned for a post-MMFF-relax organic small molecule (not a raw predicted
# pose), so tighter than that script's tolerance for raw ABCfold poses.
MAX_BOND_STRETCH = 0.4  # Angstrom beyond covalent-radius sum
MIN_NONBONDED_DIST = 1.6  # Angstrom, any non-bonded heavy-atom pair


def fetch_ccd_cif(ccd_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{ccd_id}.cif"
    if not out_path.exists():
        url = CCD_DOWNLOAD_URL.format(ccd_id=ccd_id)
        urllib.request.urlretrieve(url, out_path)
    return out_path


def build_ga3_real_mol(cif_path: Path) -> Chem.Mol:
    """RDKit mol for GA3's CCD entry: real ('model') 3D coordinates +
    explicit bond orders from the CCD's own bond table, heavy atoms only
    (H stripped after sanitization, same atom-typing convention as the
    SMILES-derived GA1 mol below)."""
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    atom_tbl = block.find(["_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol",
                            "_chem_comp_atom.charge", "_chem_comp_atom.model_Cartn_x",
                            "_chem_comp_atom.model_Cartn_y", "_chem_comp_atom.model_Cartn_z"])
    bond_tbl = block.find(["_chem_comp_bond.atom_id_1", "_chem_comp_bond.atom_id_2",
                            "_chem_comp_bond.value_order"])

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atom_tbl))
    name_to_idx: dict[str, int] = {}
    for i, row in enumerate(atom_tbl):
        atom_id, symbol, charge, x, y, z = row
        a = Chem.Atom(symbol)
        a.SetFormalCharge(int(charge))
        a.SetNoImplicit(False)
        idx = mol.AddAtom(a)
        name_to_idx[atom_id] = idx
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))

    for atom_id_1, atom_id_2, value_order in bond_tbl:
        mol.AddBond(name_to_idx[atom_id_1], name_to_idx[atom_id_2], _BOND_ORDER[value_order])

    mol.AddConformer(conf, assignId=True)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def build_ga1_mol() -> Chem.Mol:
    smiles = config.load_ligand_smiles(config.LIGAND_KEY)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"config.yaml GA1 SMILES failed to parse: {smiles!r}")
    return mol


def find_mcs(ga3_real: Chem.Mol, ga1: Chem.Mol) -> rdFMCS.MCSResult:
    result = rdFMCS.FindMCS(
        [ga3_real, ga1],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=60,
    )
    if result.canceled:
        raise RuntimeError("MCS search timed out/canceled -- GA1/GA3 should share most of their scaffold; "
                            "investigate before trusting any embed built from this.")
    return result


def constrained_embed_ga1(ga3_real: Chem.Mol, ga1: Chem.Mol, mcs: rdFMCS.MCSResult) -> tuple[Chem.Mol, float]:
    """Embed GA1 with its MCS-matched atoms tethered to GA3's real
    coordinates (AllChem.ConstrainedEmbed), MMFF-relaxing everything else
    freely. Returns (embedded mol with explicit Hs, tether RMSD in
    Angstrom -- how far the matched atoms moved from GA3's real
    positions)."""
    query = Chem.MolFromSmarts(mcs.smartsString)
    ga3_match = ga3_real.GetSubstructMatch(query)
    if not ga3_match:
        raise RuntimeError("MCS SMARTS didn't match back onto the GA3 template mol -- MCS search result is inconsistent.")

    core = Chem.RWMol(query)
    core_conf = Chem.Conformer(core.GetNumAtoms())
    ga3_conf = ga3_real.GetConformer()
    for core_idx, ga3_idx in enumerate(ga3_match):
        core_conf.SetAtomPosition(core_idx, ga3_conf.GetAtomPosition(ga3_idx))
    core = core.GetMol()
    core.AddConformer(core_conf, assignId=True)

    ga1_h = Chem.AddHs(Chem.Mol(ga1))
    AllChem.ConstrainedEmbed(ga1_h, core, useTethers=True, randomseed=42)

    ga1_match = ga1_h.GetSubstructMatch(query)
    ga1_conf = ga1_h.GetConformer()
    sq_dev = 0.0
    for core_idx, ga1_idx in enumerate(ga1_match):
        p_core = core_conf.GetAtomPosition(core_idx)
        p_ga1 = ga1_conf.GetAtomPosition(ga1_idx)
        sq_dev += (p_core - p_ga1).LengthSq()
    tether_rmsd = (sq_dev / len(ga1_match)) ** 0.5

    return ga1_h, tether_rmsd


def sanity_check(mol: Chem.Mol) -> list[str]:
    """Bond-length-vs-covalent-radii and non-bonded-clash check, same
    category of guard as sanitize_for_chimerax.py's for raw ABCfold poses
    -- returns a list of problem descriptions (empty = clean)."""
    pt = Chem.GetPeriodicTable()
    conf = mol.GetConformer()
    problems = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = (conf.GetAtomPosition(i) - conf.GetAtomPosition(j)).Length()
        expected = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) + \
            pt.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
        if dist > expected + MAX_BOND_STRETCH:
            problems.append(f"bond {mol.GetAtomWithIdx(i).GetSymbol()}{i}-{mol.GetAtomWithIdx(j).GetSymbol()}{j}: "
                             f"{dist:.2f} A vs expected ~{expected:.2f} A")

    bonded_pairs = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    for a_i, i in enumerate(heavy_idx):
        for j in heavy_idx[a_i + 1:]:
            if frozenset((i, j)) in bonded_pairs:
                continue
            dist = (conf.GetAtomPosition(i) - conf.GetAtomPosition(j)).Length()
            if dist < MIN_NONBONDED_DIST:
                problems.append(f"non-bonded clash {mol.GetAtomWithIdx(i).GetSymbol()}{i}-"
                                 f"{mol.GetAtomWithIdx(j).GetSymbol()}{j}: {dist:.2f} A")
    return problems


def main() -> None:
    ga3_cif = fetch_ccd_cif(config.GA3_CCD_ID)
    ga3_real = build_ga3_real_mol(ga3_cif)
    ga1 = build_ga1_mol()

    mcs = find_mcs(ga3_real, ga1)
    ga1_embedded, tether_rmsd = constrained_embed_ga1(ga3_real, ga1, mcs)
    problems = sanity_check(ga1_embedded)

    writer = Chem.SDWriter(str(config.GA1_FROM_GA3_SDF))
    ga1_embedded.SetProp("_Name", "GA1_from_GA3")
    writer.write(ga1_embedded)
    writer.close()

    n_ga1_heavy = ga1.GetNumAtoms()
    coverage_pct = 100.0 * mcs.numAtoms / n_ga1_heavy
    log_lines = [
        f"GA3 CCD source: {ga3_cif}",
        f"GA1 SMILES: {config.load_ligand_smiles(config.LIGAND_KEY)}",
        f"MCS: {mcs.numAtoms}/{n_ga1_heavy} GA1 heavy atoms matched ({coverage_pct:.1f}%), "
        f"{mcs.numBonds} bonds, SMARTS: {mcs.smartsString}",
        f"Tether RMSD (matched atoms vs. GA3 real coords): {tether_rmsd:.3f} A",
        f"Sanity check: {'CLEAN' if not problems else 'PROBLEMS FOUND'}",
    ]
    log_lines.extend(f"  - {p}" for p in problems)
    config.GA1_FROM_GA3_LOG.write_text("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
