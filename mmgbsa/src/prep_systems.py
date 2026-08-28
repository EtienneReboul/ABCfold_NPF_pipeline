"""
mmgbsa/src/prep_systems.py
==========================
Stage 2: build a solvated GROMACS system for each manifest complex, ready for
the Stage 3 MD. CPU-only -- run on a login node or the `fast` partition, not
the GPU array.

Per complex (results/systems/<complex_id>/):
  1. Split the HADDOCK3 pose (flexref_<N>.pdb.gz) into protein (chain A) and
     GA1 (chain B).
  2. Place the acpype GA1 conformer -- which is in the exact atom order of
     data/ligand_params/GA1_GMX.itp -- onto the pose's GA1 by an rdFMCS
     substructure match + rigid AlignMol (same idea as
     redocking/src/build_ga1_from_ga3.py). This sidesteps any CNS atom-name /
     ordering drift in the docked ligand: we keep OUR topology's atoms and
     just move them to the docked position.
  3. gmx pdb2gmx (amber99sb-ildn / tip3p) on the protein.
  4. Merge ligand into topol.top (#include GA1.itp after the ff include; add
     `GA1 1` to [ molecules ]) and into the coordinate file.
  5. editconf (dodecahedron, 1.2 nm) -> solvate (tip3p) -> genion (neutral +
     0.15 M NaCl).
  6. genrestr -> posre_ca.itp (C-alpha, 100 kJ/mol/nm2) and posre_lig.itp
     (ligand heavy atoms, 1000) ; wire both into the topology with their own
     #ifdef guards (POSRES_CA / POSRES_LIG). pdb2gmx's own POSRES/posre.itp
     block is left as-is.
  7. make_ndx -> index.ndx with a merged `Protein_GA1` group (tc-grps in the
     mdp files) and a bare `GA1` group (gmx_MMPBSA -cg ligand selection).
  8. Copy in em/nvt/npt mdp files (prod.mdp is written per-replica in Stage 3).

Idempotent: skips any complex whose results/systems/<id>/prep.done exists,
unless --force.

Usage:
    python prep_systems.py [--complex-id ID] [--limit N] [--smoke] [--force]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config
import mdp_templates as mdp

GMX = shutil.which("gmx") or "gmx"


def gmx(args: list[str], cwd: Path, inp: str | None = None, log: Path | None = None) -> None:
    cmd = [GMX, *args]
    proc = subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, text=True)
    if log is not None:
        with log.open("a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"gmx {args[0]} failed (exit {proc.returncode}) in {cwd}")


# --------------------------------------------------------------------------- #
# pose splitting + ligand placement
# --------------------------------------------------------------------------- #
def split_pose(pose_pdb: Path, out_protein: Path, out_ligand: Path) -> None:
    """Write protein (chain A ATOM) and ligand (chain B / resname GA1) PDBs."""
    prot, lig = [], []
    with config.open_maybe_gzip(pose_pdb) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21]
            resn = line[17:20].strip()
            if resn == config.LIGAND_RESNAME or chain == config.LIGAND_CHAIN_HADDOCK:
                lig.append(line)
            elif chain == config.PROTEIN_CHAIN:
                prot.append(line)
    if not prot:
        raise RuntimeError(f"{pose_pdb}: no chain {config.PROTEIN_CHAIN} protein atoms")
    if not lig:
        raise RuntimeError(f"{pose_pdb}: no chain {config.LIGAND_CHAIN_HADDOCK}/{config.LIGAND_RESNAME} ligand atoms")
    out_protein.write_text("".join(prot) + "END\n")
    out_ligand.write_text("".join(lig) + "END\n")


def place_ligand_conformer(pose_ligand_pdb: Path, out_pdb: Path) -> None:
    """Move the GA1 template onto the docked ligand by MCS + rigid align, then
    write it as PDB (resname GA1, chain B).

    Template source is the original build_ga1_from_ga3.sdf, NOT acpype's
    GA1.mol2: acpype writes GAFF atom types (c3/ca/os/hc/...) in the mol2
    element column, which RDKit's mol2 reader rejects ("Element 'c3' not
    found"). The SDF is the same 49-atom molecule in the same atom order
    acpype preserved into GA1_GMX.itp, with real element symbols + bonds."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS

    templ = Chem.MolFromMolFile(str(config.GA1_SDF), removeHs=False, sanitize=True)
    if templ is None:
        raise RuntimeError(f"could not read {config.GA1_SDF} (GA1 template)")

    block = pose_ligand_pdb.read_text()
    pose = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    if pose is None:
        raise RuntimeError(f"RDKit could not parse {pose_ligand_pdb}")
    try:
        Chem.SanitizeMol(pose)
    except Exception:
        pass  # docked ligand geometry can trip valence perception; MCS on connectivity still works

    mcs = rdFMCS.FindMCS([templ, pose], atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True,
                         completeRingsOnly=False, timeout=30)
    if mcs.numAtoms < 12:
        raise RuntimeError(f"{pose_ligand_pdb}: MCS with GA1 template only {mcs.numAtoms} atoms -- pose ligand suspect")
    patt = Chem.MolFromSmarts(mcs.smartsString)
    tmatch = templ.GetSubstructMatch(patt)
    pmatch = pose.GetSubstructMatch(patt)
    if len(tmatch) != len(pmatch) or not tmatch:
        raise RuntimeError(f"{pose_ligand_pdb}: MCS atom map mismatch ({len(tmatch)} vs {len(pmatch)})")

    rmsd = AllChem.AlignMol(templ, pose, atomMap=list(zip(tmatch, pmatch)))
    print(f"      ligand align: MCS {mcs.numAtoms} atoms, heavy-atom RMSD {rmsd:.3f} A")

    # keep GA1 resname / chain B so downstream make_ndx `r GA1` works
    for atom in templ.GetAtoms():
        mi = atom.GetPDBResidueInfo()
        if mi is None:
            mi = Chem.AtomPDBResidueInfo()
        mi.SetResidueName("GA1 ")
        mi.SetResidueNumber(1)
        mi.SetChainId("B")
        mi.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(mi)
    Chem.MolToPDBFile(templ, str(out_pdb))


# --------------------------------------------------------------------------- #
# topology surgery
# --------------------------------------------------------------------------- #
def splice_ligand_into_topology(topol: Path) -> None:
    """Add `#include "GA1.itp"` right after the forcefield include (so the
    ligand's [ atomtypes ] precede every [ moleculetype ]) and append
    `GA1 1` to [ molecules ] (the last section pdb2gmx writes)."""
    lines = topol.read_text().splitlines()
    out: list[str] = []
    added_include = False
    seen_molecules = False
    for line in lines:
        out.append(line)
        if not added_include and line.strip().startswith('#include') and 'forcefield.itp' in line:
            out += ['', '; ---- GA1 ligand (mmgbsa/src/prep_systems.py) ----',
                    '#include "GA1.itp"',
                    '#ifdef POSRES_LIG', '#include "posre_lig.itp"', '#endif', '']
            added_include = True
        if line.strip().lower().replace(' ', '') == '[molecules]':
            seen_molecules = True
    if not (added_include and seen_molecules):
        raise RuntimeError(f"topology surgery failed on {topol} "
                           f"(ff-include found={added_include}, [molecules] found={seen_molecules})")
    out.append(f'{config.LIGAND_RESNAME}                 1')
    topol.write_text("\n".join(out) + "\n")


def wire_ca_posres(topol: Path) -> None:
    """Add a POSRES_CA-guarded include of posre_ca.itp right after pdb2gmx's
    own `#ifdef POSRES ... #include "posre.itp" ... #endif` block in the
    protein moleculetype."""
    lines = topol.read_text().splitlines()
    out: list[str] = []
    i = 0
    wired = False
    while i < len(lines):
        out.append(lines[i])
        if (not wired and lines[i].strip() == '#ifdef POSRES'
                and i + 2 < len(lines) and 'posre.itp' in lines[i + 1]):
            out.append(lines[i + 1])          # #include "posre.itp"
            out.append(lines[i + 2])          # #endif
            out += ['#ifdef POSRES_CA', '#include "posre_ca.itp"', '#endif']
            i += 3
            wired = True
            continue
        i += 1
    if not wired:
        raise RuntimeError(f"could not find pdb2gmx POSRES block in {topol} to attach POSRES_CA")
    topol.write_text("\n".join(out) + "\n")


_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
    "CYX", "CYM", "ASH", "GLH", "LYN", "ACE", "NME", "NMA",
}
_WATER = {"SOL", "HOH", "WAT", "TIP3", "T3P"}
_IONS = {"NA", "CL", "K", "MG", "CA", "ZN", "NA+", "CL-", "SOD", "CLA"}


def write_index_ndx(system_gro: Path, out_ndx: Path) -> None:
    """Deterministic index.ndx from a .gro: System / Protein / GA1 /
    Water / Ion / Protein_GA1 / Water_and_ions. Atom serials are 1-based,
    matching GROMACS .gro atom order."""
    lines = system_gro.read_text().splitlines()
    natoms = int(lines[1])
    groups: dict[str, list[int]] = {k: [] for k in
                                    ("System", "Protein", config.LIGAND_RESNAME, "Water", "Ion")}
    for i, line in enumerate(lines[2:2 + natoms], start=1):
        resname = line[5:10].strip()
        groups["System"].append(i)
        if resname in _AA3:
            groups["Protein"].append(i)
        elif resname == config.LIGAND_RESNAME or resname == "MOL":
            groups[config.LIGAND_RESNAME].append(i)
        elif resname in _WATER:
            groups["Water"].append(i)
        elif resname in _IONS:
            groups["Ion"].append(i)
    groups["Protein_GA1"] = sorted(groups["Protein"] + groups[config.LIGAND_RESNAME])
    groups["Water_and_ions"] = sorted(groups["Water"] + groups["Ion"])

    if not groups[config.LIGAND_RESNAME]:
        raise RuntimeError(f"{system_gro}: no ligand atoms (resname {config.LIGAND_RESNAME}/MOL) found")
    if not groups["Protein"]:
        raise RuntimeError(f"{system_gro}: no protein atoms found")

    with out_ndx.open("w") as fh:
        for name, idxs in groups.items():
            fh.write(f"[ {name} ]\n")
            for j in range(0, len(idxs), 15):
                fh.write(" ".join(f"{k:>7d}" for k in idxs[j:j + 15]) + "\n")
            fh.write("\n")


def merge_gro(protein_gro: Path, ligand_gro: Path, out_gro: Path) -> None:
    p = protein_gro.read_text().splitlines()
    l = ligand_gro.read_text().splitlines()
    p_n = int(p[1].strip())
    l_n = int(l[1].strip())
    body = p[2:2 + p_n] + l[2:2 + l_n]
    box = p[2 + p_n]
    out_gro.write_text(p[0] + "\n" + str(p_n + l_n) + "\n" + "\n".join(body) + "\n" + box + "\n")


# --------------------------------------------------------------------------- #
# per-complex driver
# --------------------------------------------------------------------------- #
def prepare_one(row: dict, smoke: bool, force: bool) -> str:
    cid = row["complex_id"]
    sysdir = config.SYSTEMS_DIR / cid
    done = sysdir / "prep.done"
    if done.exists() and not force:
        return "skip (done)"
    sysdir.mkdir(parents=True, exist_ok=True)
    log = sysdir / "prep.log"
    log.write_text(f"# prep_systems.py {cid}\n")

    pose = config.PIPELINE_ROOT / row["pose_pdb"]
    if not pose.exists():
        raise FileNotFoundError(pose)

    split_pose(pose, sysdir / "protein_raw.pdb", sysdir / "ligand_pose.pdb")
    place_ligand_conformer(sysdir / "ligand_pose.pdb", sysdir / "ligand.pdb")
    shutil.copy(config.LIGAND_PARAMS_DIR / "GA1_GMX.itp", sysdir / "GA1.itp")

    # 3. pdb2gmx on the protein
    gmx(["pdb2gmx", "-f", "protein_raw.pdb", "-o", "protein.gro", "-p", "topol.top",
         "-i", "posre.itp", "-ff", config.FORCEFIELD_PROTEIN, "-water", config.WATER_MODEL,
         "-ignh"], cwd=sysdir, log=log)

    # 4. ligand -> .gro, merge coords + topology
    gmx(["editconf", "-f", "ligand.pdb", "-o", "ligand.gro"], cwd=sysdir, log=log)
    merge_gro(sysdir / "protein.gro", sysdir / "ligand.gro", sysdir / "complex.gro")
    splice_ligand_into_topology(sysdir / "topol.top")

    # 5. box / solvate / ions
    gmx(["editconf", "-f", "complex.gro", "-o", "box.gro", "-bt", "dodecahedron", "-d", "1.2", "-c"],
        cwd=sysdir, log=log)
    gmx(["solvate", "-cp", "box.gro", "-cs", "spc216.gro", "-p", "topol.top", "-o", "solv.gro"],
        cwd=sysdir, log=log)
    (sysdir / "ions.mdp").write_text(mdp.IONS)
    gmx(["grompp", "-f", "ions.mdp", "-c", "solv.gro", "-p", "topol.top", "-o", "ions.tpr",
         "-maxwarn", "2"], cwd=sysdir, log=log)
    gmx(["genion", "-s", "ions.tpr", "-o", "system.gro", "-p", "topol.top",
         "-pname", "NA", "-nname", "CL", "-neutral", "-conc", str(config.SALT_MOLAR)],
        cwd=sysdir, inp="SOL\n", log=log)

    # 6. restraints: CA (protein.gro numbering) + ligand heavy atoms
    gmx(["genrestr", "-f", "protein.gro", "-o", "posre_ca.itp",
         "-fc", str(config.CA_POSRES_KJ), str(config.CA_POSRES_KJ), str(config.CA_POSRES_KJ)],
        cwd=sysdir, inp="C-alpha\n", log=log)
    gmx(["genrestr", "-f", "ligand.gro", "-o", "posre_lig.itp", "-fc", "1000", "1000", "1000"],
        cwd=sysdir, inp="0\n", log=log)  # group 0 = System = the ligand alone in ligand.gro
    wire_ca_posres(sysdir / "topol.top")

    # 7. index groups -- built directly from system.gro (deterministic; avoids
    #    gmx make_ndx's shifting default-group numbering). Needs: GA1
    #    (gmx_MMPBSA ligand selection), Protein_GA1 and Water_and_ions (mdp
    #    tc-grps).
    write_index_ndx(sysdir / "system.gro", sysdir / "index.ndx")

    # 8. equilibration mdp files (prod.mdp is per-replica, Stage 3)
    (sysdir / "em.mdp").write_text(mdp.EM)
    (sysdir / "nvt.mdp").write_text(mdp.NVT)
    (sysdir / "npt.mdp").write_text(mdp.NPT)
    if smoke:
        (sysdir / "SMOKE").write_text("prod shortened to 100 ps in Stage 3\n")

    done.write_text("ok\n")
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complex-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true",
                    help="tag systems for the shortened-production smoke test (Verification step 1)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = config.read_csv_rows(config.MANIFEST_CSV)
    if args.complex_id:
        rows = [r for r in rows if r["complex_id"] == args.complex_id]
    elif args.smoke:
        rows = config.smoke_rows(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("[stage2] no matching manifest rows"); return

    for r in rows:
        cid = r["complex_id"]
        try:
            status = prepare_one(r, smoke=args.smoke, force=args.force)
            print(f"[stage2] {cid}: {status}")
        except Exception as exc:  # keep going -- one bad pose shouldn't stall the batch
            print(f"[stage2] {cid}: FAILED -- {exc}")


if __name__ == "__main__":
    main()
