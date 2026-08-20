#!/usr/bin/env python3
"""
rescoring/src/plip_run_batch.py
===================================
Runs PLIP (Protein-Ligand Interaction Profiler, via Docker --
pharmai/plip:latest) on every ChimeraX-minimized complex
(chimerax_run_batch.py's output), producing one *_report.txt per complex
under results/plip/.

Batched, not one docker invocation per complex: PLIP's own `-f` flag
accepts multiple input files plus `--maxthreads N` to process them
concurrently INSIDE one container -- confirmed by hand this avoids
per-complex Docker container startup overhead (would otherwise dominate
wall time across 16k+ complexes) while still using multiple cores. Chunked
into batches of --batch-size files (not all 16k+ in one invocation) to
keep each docker argv well under the OS ARG_MAX and to get periodic
progress/resumability checkpoints.

The minimized PDB already carries explicit hydrogens from ChimeraX's own
dock-prep + minimize step (see chimerax_minimize_pose.py) -- `--nohydro`
tells PLIP not to re-add polar hydrogens on top of those.

PLIP auto-detects our ligand's HETATM records (chain L, one of
config.LIGAND_CODES's synthetic 3-letter codes) as a small-molecule ligand
with no extra flags needed -- confirmed by hand on a real complex.

Usage:
    python plip_run_batch.py --batch-size 300 --maxthreads 8
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

import config

PLIP_DIR = config.RESULTS_DIR / "plip"
SCRATCH_DIR = config.RESULTS_DIR / "plip_scratch"
MINIMIZED_DIR = config.RESULTS_DIR / "chimerax_minimized"

DEFAULT_IMAGE = "pharmai/plip:latest"
DEFAULT_PLATFORM = "linux/amd64"


def pending_complex_ids() -> list[str]:
    manifest = pd.read_csv(config.MANIFEST_CSV)
    out = []
    for cid in manifest["complex_id"]:
        if not (MINIMIZED_DIR / f"{cid}.pdb").exists():
            continue  # no minimized pose yet -- chimerax_run_batch.py hasn't gotten to it
        if (PLIP_DIR / f"{cid}_report.txt").exists():
            continue  # already done
        out.append(cid)
    return out


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_batch(complex_ids: list[str], image: str, platform: str, maxthreads: int) -> None:
    cmd = ["docker", "run", "--rm"]
    if platform:
        cmd += ["--platform", platform]
    cmd += [
        "-v", f"{MINIMIZED_DIR}:/in:ro",
        "-v", f"{SCRATCH_DIR}:/out",
        image,
        "-f", *[f"/in/{cid}.pdb" for cid in complex_ids],
        "-t", "-o", "/out",
        "--maxthreads", str(maxthreads),
        "--nohydro", "--nofixfile", "--quiet",
    ]
    subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3600)

    PLIP_DIR.mkdir(parents=True, exist_ok=True)
    for report in SCRATCH_DIR.glob("*_report.txt"):
        report.replace(PLIP_DIR / report.name)
    # Everything else PLIP wrote to /out (protonated.pdb, any leftovers) is
    # scratch -- clear it so it doesn't accumulate across batches.
    for leftover in SCRATCH_DIR.iterdir():
        if leftover.is_file():
            leftover.unlink()
        else:
            shutil.rmtree(leftover)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--maxthreads", type=int, default=8)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--platform", default=DEFAULT_PLATFORM)
    args = ap.parse_args()

    PLIP_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    todo = pending_complex_ids()
    if not todo:
        print("[plip_run_batch] nothing to do -- every minimized complex already has a PLIP report")
        return
    print(f"[plip_run_batch] {len(todo)} complex(es) pending, batch size {args.batch_size}, "
          f"--maxthreads {args.maxthreads}")

    t0 = time.time()
    n_done = 0
    batches = list(chunks(todo, args.batch_size))
    for i, batch in enumerate(batches, 1):
        run_batch(batch, args.image, args.platform, args.maxthreads)
        n_ok = sum((PLIP_DIR / f"{cid}_report.txt").exists() for cid in batch)
        n_done += len(batch)
        elapsed = time.time() - t0
        rate = n_done / elapsed
        eta = (len(todo) - n_done) / rate if rate > 0 else float("nan")
        print(f"[plip_run_batch] batch {i}/{len(batches)}: {n_ok}/{len(batch)} reports written "
              f"({n_done}/{len(todo)} total, {elapsed:.0f}s elapsed, ETA {eta / 60:.0f} min)")

    n_missing = len(pending_complex_ids())
    if n_missing:
        print(f"[plip_run_batch] WARNING: {n_missing} complex(es) still have no report after this run "
              "(PLIP found no ligand/failed on them -- see individual logs if needed)")
    print(f"[plip_run_batch] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
