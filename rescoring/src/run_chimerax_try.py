#!/usr/bin/env python3
"""
rescoring/src/run_chimerax_try.py
=====================================
One-shot driver to try ChimeraX's own `minimize` command as an alternative
to the PyRosetta FastRelax stage (relief.py), for a single complex at a
time. Standalone / experimental -- not wired into
worflows/postprocessing/Snakefile. Two sub-steps, each in its own file
because each needs a different Python environment:

  1. sanitize_for_chimerax.py -- runs in this repo's normal env (rdkit,
     gemmi, pandas). Stages the pose with SMILES-corrected ligand chemistry
     and a ligand-geometry sanity check, raising instead of handing a
     broken pose to the minimizer.
  2. chimerax_minimize_pose.py -- runs *inside* ChimeraX itself (`chimerax
     --nogui --offscreen --script`), since `minimize`/`session` only exist
     in a live ChimeraX process; invoked here as a subprocess.

Usage:
    python run_chimerax_try.py --complex-id <id> [--chimerax /path/to/ChimeraX]

Writes results/chimerax_try/<complex_id>_staged.pdb (sanitized input) and
results/chimerax_try/<complex_id>_minimized.pdb (+ _energy.csv).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import config
from sanitize_for_chimerax import LigandGeometryError, sanitize_and_stage

OUT_DIR = config.RESULTS_DIR / "chimerax_try"
SCRIPT_DIR = Path(__file__).resolve().parent

# The `chimerax` shell alias many users have (as here) is a zsh alias, not
# an executable on PATH -- subprocess needs the real binary path. Falls
# back to plain "chimerax" (works if it genuinely is on PATH) when this
# default doesn't exist.
DEFAULT_CHIMERAX = "/Applications/ChimeraX_Daily.app/Contents/MacOS/ChimeraX"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complex-id", required=True)
    ap.add_argument("--chimerax", default=None,
                     help=f"ChimeraX executable path (default: {DEFAULT_CHIMERAX} if it exists, else 'chimerax')")
    args = ap.parse_args()

    chimerax_bin = args.chimerax
    if chimerax_bin is None:
        chimerax_bin = DEFAULT_CHIMERAX if Path(DEFAULT_CHIMERAX).exists() else "chimerax"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    staged_pdb = OUT_DIR / f"{args.complex_id}_staged.pdb"
    minimized_pdb = OUT_DIR / f"{args.complex_id}_minimized.pdb"

    try:
        sanitize_and_stage(args.complex_id, staged_pdb)
    except LigandGeometryError as e:
        sys.exit(f"[run_chimerax_try] {e}")
    print(f"[run_chimerax_try] staged -> {staged_pdb}")

    script_path = SCRIPT_DIR / "chimerax_minimize_pose.py"
    cmd = [chimerax_bin, "--nogui", "--offscreen", "--script",
           f"{script_path} {staged_pdb} {minimized_pdb}"]
    print(f"[run_chimerax_try] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[run_chimerax_try] minimized -> {minimized_pdb}")


if __name__ == "__main__":
    main()
