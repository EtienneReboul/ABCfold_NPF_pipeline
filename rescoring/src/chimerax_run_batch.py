#!/usr/bin/env python3
"""
rescoring/src/chimerax_run_batch.py
=======================================
Production batch driver for the ChimeraX minimization path (see
run_chimerax_try.py / rescoring/README.md's "Trying ChimeraX minimization"
section for the experimental single-complex version this scales up).

Unlike PyRosetta's staged_poses (deleted right after scoring -- pure
scratch, see run_complex.py), the minimized PDB here IS the product: PLIP
runs directly on it downstream (plip_run_batch.py), so every
results/chimerax_minimized/<complex_id>.pdb is kept permanently. Only the
intermediate staged (pre-minimization) PDB is scratch and gets deleted.

Runs `--workers` ChimeraX subprocesses concurrently via a thread pool (not
a process pool -- each unit of work is dominated by a subprocess.run() call
that releases the GIL while ChimeraX runs, so threads are enough and avoid
pickling/fork overhead for no benefit). Resumable: skips any complex_id
that already has a minimized PDB on disk.

Usage:
    python chimerax_run_batch.py --workers 8 [--protein NPF2.14_Q9CAR9]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import config
from sanitize_for_chimerax import LigandGeometryError, sanitize_and_stage

MINIMIZED_DIR = config.RESULTS_DIR / "chimerax_minimized"
STAGED_SCRATCH_DIR = config.RESULTS_DIR / "chimerax_staged_scratch"
SCRIPT_DIR = Path(__file__).resolve().parent
CHIMERAX_SCRIPT = SCRIPT_DIR / "chimerax_minimize_pose.py"
DEFAULT_CHIMERAX = "/Applications/ChimeraX_Daily.app/Contents/MacOS/ChimeraX"


def pending_complex_ids(protein: str | None) -> list[str]:
    manifest = pd.read_csv(config.MANIFEST_CSV)
    if protein:
        manifest = manifest[manifest["protein"] == protein]
    all_ids = manifest["complex_id"].tolist()
    return [cid for cid in all_ids if not (MINIMIZED_DIR / f"{cid}.pdb").exists()]


def minimize_one(complex_id: str, chimerax_bin: str) -> tuple[str, str, str]:
    """Returns (complex_id, status, detail)."""
    staged_pdb = STAGED_SCRATCH_DIR / f"{complex_id}.pdb"
    minimized_pdb = MINIMIZED_DIR / f"{complex_id}.pdb"
    try:
        try:
            sanitize_and_stage(complex_id, staged_pdb)
        except LigandGeometryError as e:
            return complex_id, "geometry_rejected", str(e)

        cmd = [chimerax_bin, "--nogui", "--offscreen", "--script",
               f"{CHIMERAX_SCRIPT} {staged_pdb} {minimized_pdb}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            return complex_id, "chimerax_failed", e.stderr[-2000:]
        except subprocess.TimeoutExpired:
            return complex_id, "chimerax_timeout", ""

        if not minimized_pdb.exists():
            return complex_id, "no_output", "ChimeraX exited 0 but wrote no output PDB"
        return complex_id, "ok", ""
    finally:
        staged_pdb.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--protein", default=None, help="restrict to one base protein name (default: all)")
    ap.add_argument("--chimerax", default=None)
    args = ap.parse_args()

    chimerax_bin = args.chimerax
    if chimerax_bin is None:
        chimerax_bin = DEFAULT_CHIMERAX if Path(DEFAULT_CHIMERAX).exists() else "chimerax"

    MINIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    todo = pending_complex_ids(args.protein)
    if not todo:
        print("[chimerax_run_batch] nothing to do -- every complex already has a minimized PDB")
        return
    print(f"[chimerax_run_batch] {len(todo)} complex(es) pending, {args.workers} workers")

    t0 = time.time()
    counts: dict[str, int] = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(minimize_one, cid, chimerax_bin): cid for cid in todo}
        for fut in as_completed(futures):
            complex_id, status, detail = fut.result()
            counts[status] = counts.get(status, 0) + 1
            n_done += 1
            if status != "ok":
                print(f"[chimerax_run_batch] {complex_id}: {status} -- {detail[:300]}", file=sys.stderr)
            if n_done % 50 == 0 or n_done == len(todo):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(todo) - n_done) / rate if rate > 0 else float("nan")
                print(f"[chimerax_run_batch] {n_done}/{len(todo)} done "
                      f"({elapsed:.0f}s elapsed, {rate:.2f}/s, ETA {eta / 60:.0f} min) -- {counts}")

    print(f"[chimerax_run_batch] finished: {counts}, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
