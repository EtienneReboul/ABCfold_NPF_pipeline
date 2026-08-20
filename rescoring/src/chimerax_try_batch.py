#!/usr/bin/env python3
"""
rescoring/src/chimerax_try_batch.py
=======================================
Ad-hoc benchmark: run the experimental ChimeraX minimization path
(sanitize_for_chimerax.py + chimerax_minimize_pose.py, see run_chimerax_try.py)
on the top-N ipTM complexes of one protein, timing each step and measuring
how much the minimization actually moved the structure (heavy-atom RMSD,
protein-only vs ligand-only, staged vs minimized) plus the ChimeraX energy
drop -- so a full-corpus ETA can be extrapolated from real per-complex cost
instead of guessed.

Not a permanent pipeline stage -- ad-hoc, run by hand, writes its own
summary CSV under results/chimerax_try/.

Usage:
    python chimerax_try_batch.py --protein NPF2.14_Q9CAR9 --top-n 20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import gemmi
import pandas as pd

import config
from sanitize_for_chimerax import LigandGeometryError, sanitize_and_stage

OUT_DIR = config.RESULTS_DIR / "chimerax_try"
SCRIPT_DIR = Path(__file__).resolve().parent
CHIMERAX_SCRIPT = SCRIPT_DIR / "chimerax_minimize_pose.py"
DEFAULT_CHIMERAX = "/Applications/ChimeraX_Daily.app/Contents/MacOS/ChimeraX"


def top_n_complex_ids(protein: str, n: int) -> pd.DataFrame:
    """Top-n manifest rows for `protein` (base name, no __holo) ranked by
    ipTM, joining manifest.csv (this pipeline's complex enumeration) back
    to meta.parquet (the only place ipTM lives) via frame_id."""
    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows = manifest[manifest["protein"] == protein].copy()
    if rows.empty:
        sys.exit(f"no manifest rows for protein={protein!r}")

    meta_path = config.ALIGN_ROOT / f"{protein}__holo" / "meta.parquet"
    meta = pd.read_parquet(meta_path)
    meta["complex_id"] = meta["protein"] + "_" + meta["frame_id"]

    merged = rows.merge(meta[["complex_id", "iptm"]], on="complex_id", how="left")
    merged = merged.sort_values("iptm", ascending=False)
    return merged.head(n)


def heavy_atom_positions(pdb_path: Path) -> dict[tuple[str, int, str], gemmi.Position]:
    """(chain, resnum, atom_name) -> position, heavy atoms only. Dock-prep
    inside ChimeraX adds hydrogens (confirmed: atom count roughly doubles),
    so staged vs. minimized files can't be compared by serial/atom-index --
    matching by (chain, resnum, name) instead, since dock-prep doesn't
    rename or reorder the atoms that already existed."""
    st = gemmi.read_structure(str(pdb_path))
    out = {}
    for chain in st[0]:
        for residue in chain:
            for atom in residue:
                if atom.element.name == "H":
                    continue
                out[(chain.name, residue.seqid.num, atom.name.strip())] = atom.pos
    return out


def rmsd(staged_path: Path, minimized_path: Path, chain: str | None = None) -> tuple[float, int]:
    a = heavy_atom_positions(staged_path)
    b = heavy_atom_positions(minimized_path)
    keys = [k for k in a if k in b and (chain is None or k[0] == chain)]
    if not keys:
        return float("nan"), 0
    sq = sum(a[k].dist(b[k]) ** 2 for k in keys)
    return (sq / len(keys)) ** 0.5, len(keys)


def read_energy_trace(minimized_path: Path) -> tuple[float, float] | tuple[None, None]:
    energy_csv = minimized_path.with_name(minimized_path.stem + "_energy.csv")
    if not energy_csv.exists():
        return None, None
    df = pd.read_csv(energy_csv)
    if df.empty:
        return None, None
    return float(df["energy_kJ_mol"].iloc[0]), float(df["energy_kJ_mol"].iloc[-1])


def run_one(complex_id: str, chimerax_bin: str) -> dict:
    staged_pdb = OUT_DIR / f"{complex_id}_staged.pdb"
    minimized_pdb = OUT_DIR / f"{complex_id}_minimized.pdb"
    row = {"complex_id": complex_id}

    t0 = time.time()
    try:
        sanitize_and_stage(complex_id, staged_pdb)
    except LigandGeometryError as e:
        row.update(status="geometry_rejected", error=str(e))
        return row
    t1 = time.time()

    cmd = [chimerax_bin, "--nogui", "--offscreen", "--script",
           f"{CHIMERAX_SCRIPT} {staged_pdb} {minimized_pdb}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        row.update(status="chimerax_failed", error=e.stderr[-2000:])
        return row
    except subprocess.TimeoutExpired:
        row.update(status="chimerax_timeout")
        return row
    t2 = time.time()

    protein_rmsd, protein_n = rmsd(staged_pdb, minimized_pdb, chain=config.PROTEIN_CHAIN)
    ligand_rmsd, ligand_n = rmsd(staged_pdb, minimized_pdb, chain=config.LIGAND_CHAIN)
    e_start, e_end = read_energy_trace(minimized_pdb)

    row.update(
        status="ok",
        sanitize_seconds=round(t1 - t0, 2),
        chimerax_seconds=round(t2 - t1, 2),
        total_seconds=round(t2 - t0, 2),
        protein_heavy_atom_rmsd=protein_rmsd,
        protein_atoms_compared=protein_n,
        ligand_heavy_atom_rmsd=ligand_rmsd,
        ligand_atoms_compared=ligand_n,
        energy_start_kJ_mol=e_start,
        energy_end_kJ_mol=e_end,
    )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein", required=True, help="base protein name, e.g. NPF2.14_Q9CAR9")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--chimerax", default=None)
    args = ap.parse_args()

    chimerax_bin = args.chimerax
    if chimerax_bin is None:
        chimerax_bin = DEFAULT_CHIMERAX if Path(DEFAULT_CHIMERAX).exists() else "chimerax"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top = top_n_complex_ids(args.protein, args.top_n)
    print(f"[chimerax_try_batch] {len(top)} complexes selected for {args.protein} "
          f"(ipTM {top['iptm'].min():.2f}-{top['iptm'].max():.2f})")

    results = []
    t_batch0 = time.time()
    for i, complex_id in enumerate(top["complex_id"], 1):
        t0 = time.time()
        row = run_one(complex_id, chimerax_bin)
        dt = time.time() - t0
        print(f"[chimerax_try_batch] ({i}/{len(top)}) {complex_id}: {row['status']} in {dt:.1f}s")
        results.append(row)
    t_batch_total = time.time() - t_batch0

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / f"{args.protein}_top{args.top_n}_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"[chimerax_try_batch] wrote {out_csv}")
    print(f"[chimerax_try_batch] batch wall time: {t_batch_total:.1f}s for {len(top)} complexes "
          f"({t_batch_total / len(top):.1f}s/complex average)")


if __name__ == "__main__":
    main()
