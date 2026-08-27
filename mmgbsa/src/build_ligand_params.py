"""
mmgbsa/src/build_ligand_params.py
=================================
Stage 0 (one-time): GAFF2 + AM1-BCC parameters for GA1, in both GROMACS and
Amber form, for the explicit-solvent MD (Stage 2/3) and the gmx_MMPBSA GB
endpoint (Stage 4).

Input: redocking/data/ga1_from_ga3.sdf -- GA1's 3D structure, RDKit-embedded
tethered to GA3's real RCSB CCD geometry (redocking/src/build_ga1_from_ga3.py,
49 atoms incl. H). Reused as-is, not rebuilt.

Charge state: GA1 is modelled as the -1 monoanion -- the C-6 carboxylic acid
is deprotonated at physiological pH (pKa ~4), the C-19->C-10 gamma-lactone is
neutral. This is a single fixed modelling choice, documented here, NOT scanned
(mmgbsa/README.md "Limitations"). If a neutral-COOH comparison is ever wanted
it's a separate `--net-charge 0` run into a separate output dir.

Route: `acpype` CLI directly (not biobb's AcpypeParamsCNS wrapper that
redocking/src/prep_ligand_topology.py uses -- that one only emits CNS format
and hard-codes the charge method). acpype reads the .sdf directly (via its
bundled OpenBabel), runs antechamber/sqm for AM1-BCC, and writes a
`GA1.acpype/` dir with GROMACS (.itp/.gro/.top), Amber (.prmtop/.inpcrd),
mol2 and frcmod all at once.

Requires: acpype + AmberTools (antechamber/sqm) on PATH -- envs/mmgbsa.yaml.

Output (mmgbsa/data/ligand_params/):
  GA1_GMX.itp        GROMACS moleculetype (spliced into each system's topol.top, Stage 2)
  GA1_GMX.gro        GROMACS coordinates (ligand alone, reference)
  GA1.mol2           GAFF2-typed, AM1-BCC-charged
  GA1_AC.frcmod      any missing GAFF2 parameters antechamber had to estimate
  GA1.prmtop/.inpcrd Amber topology (ligand-alone leg / cross-checks)
  GA1_params.json    provenance: net charge, atom count, per-atom charge sum, acpype version
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import config


def run_acpype(sdf_path: Path, work_dir: Path, net_charge: int, basename: str = "GA1") -> Path:
    """acpype CLI -> `<work_dir>/<basename>.acpype/`. Returns that dir."""
    acpype = shutil.which("acpype")
    if acpype is None:
        raise RuntimeError("acpype not found on PATH -- activate the mmgbsa conda env (envs/mmgbsa.yaml).")
    work_dir.mkdir(parents=True, exist_ok=True)
    # -c bcc  : AM1-BCC charges (antechamber/sqm)
    # -a gaff2 : GAFF2 atom types
    # -n <q>  : net charge (acpype does NOT reliably infer a formal charge from an SDF)
    # -b      : basename for every output file
    cmd = [acpype, "-i", str(sdf_path.resolve()), "-b", basename,
           "-n", str(net_charge), "-a", "gaff2", "-c", "bcc", "-o", "gmx"]
    print("[stage0]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"acpype failed (exit {proc.returncode}) -- see output above.")
    out_dir = work_dir / f"{basename}.acpype"
    if not out_dir.is_dir():
        raise FileNotFoundError(f"acpype reported success but {out_dir} is missing.")
    return out_dir


def _itp_charge_sum_and_count(itp_path: Path) -> tuple[float, int]:
    """Sum column 7 (charge) and count rows of an .itp [ atoms ] section."""
    lines = itp_path.read_text().splitlines()
    in_atoms = False
    total = 0.0
    n = 0
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            in_atoms = s.replace(" ", "").lower() == "[atoms]"
            continue
        if not in_atoms or not s or s.startswith(";"):
            continue
        cols = s.split()
        if len(cols) < 7:
            continue
        total += float(cols[6])
        n += 1
    return total, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-charge", type=int, default=config.LIGAND_NET_CHARGE,
                    help=f"formal charge of the modelled ligand (default {config.LIGAND_NET_CHARGE})")
    ap.add_argument("--sdf", type=Path, default=config.GA1_SDF)
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs already exist")
    args = ap.parse_args()

    out_itp = config.LIGAND_PARAMS_DIR / "GA1_GMX.itp"
    if out_itp.exists() and not args.force:
        print(f"[stage0] {out_itp} already exists -- pass --force to rebuild. Nothing to do.")
        return

    if not args.sdf.exists():
        raise FileNotFoundError(f"{args.sdf} not found -- run redocking/src/build_ga1_from_ga3.py first.")

    work_dir = config.DATA_DIR / "_cache" / "acpype"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    acpype_dir = run_acpype(args.sdf, work_dir, args.net_charge)

    # acpype filenames: <base>_GMX.itp / <base>_GMX.gro / <base>_GMX.top /
    # <base>.mol2 / <base>_AC.frcmod / <base>_AMBER.prmtop / <base>_AMBER.inpcrd
    copies = {
        "GA1_GMX.itp": "GA1_GMX.itp",
        "GA1_GMX.gro": "GA1_GMX.gro",
        "GA1_GMX.top": "GA1_GMX.top",
        "GA1.mol2": "GA1.mol2",
        "GA1_AC.frcmod": "GA1_AC.frcmod",
        "GA1_AMBER.prmtop": "GA1.prmtop",
        "GA1_AMBER.inpcrd": "GA1.inpcrd",
    }
    config.LIGAND_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in copies.items():
        src = acpype_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"expected acpype output {src} missing -- check {acpype_dir} log files.")
        shutil.copy(src, config.LIGAND_PARAMS_DIR / dst_name)

    charge_sum, n_atoms = _itp_charge_sum_and_count(out_itp)
    if round(charge_sum) != args.net_charge:
        raise ValueError(f"GA1_GMX.itp charge sum {charge_sum:.4f} rounds to {round(charge_sum)}, "
                         f"expected net charge {args.net_charge}.")

    acpype_ver = ""
    m = re.search(r"acpype.*?(\d+\.\d+[\w.]*)", subprocess.run(
        [shutil.which("acpype"), "-v"], capture_output=True, text=True).stdout + " ")
    if m:
        acpype_ver = m.group(1)

    prov = {
        "ligand_key": config.LIGAND_KEY,
        "source_sdf": str(args.sdf),
        "net_charge": args.net_charge,
        "atom_count": n_atoms,
        "itp_charge_sum": round(charge_sum, 5),
        "atom_types": "gaff2",
        "charge_method": "bcc (AM1-BCC)",
        "acpype_version": acpype_ver,
    }
    (config.LIGAND_PARAMS_DIR / "GA1_params.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"[stage0] OK -- {n_atoms} atoms, charge sum {charge_sum:+.4f} (net {args.net_charge:+d})")
    print(f"[stage0] wrote {config.LIGAND_PARAMS_DIR}/{{GA1_GMX.itp,GA1.mol2,GA1.prmtop,GA1_params.json,...}}")


if __name__ == "__main__":
    main()
