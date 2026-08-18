#!/usr/bin/env python3
"""
rescoring/src/run_complex.py
===============================
Run the full pipeline on ONE complex: stage the corrected pose
(pose_prep.py) -> clash relief (relief.py) -> per-residue decomposition
(decompose.py) -> tidy CSV.

run_batch.py calls `run_one()` directly (in-process, not via subprocess) to
drive the full manifest.

Usage:
    python run_complex.py --complex-id NPF2.12_Q9LFX9__holo_alphafold3_seed123_sample4 \\
        [--n-replicas 1] [--relax-cycles 1]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyrosetta

import config
import decompose as dc
import ligand_fix as lf
import pose_prep as pp
import relief as rl

STAGED_DIR = config.STAGED_DIR


def run_one(complex_id: str, cif_path: Path, protein: str, ligand: str, ligand_chain: str,
            template, ca_cluster, ligand_pose_cluster,
            n_replicas: int = 1, relax_cycles: int = 1,
            seed: int | None = None, log_file=None) -> pd.DataFrame:
    def log(msg):
        print(msg, file=log_file or sys.stdout, flush=True)

    ligand_resname = config.ligand_resname(ligand)
    t0 = time.time()
    staged_path = STAGED_DIR / f"{complex_id}.pdb"
    pp.prepare_complex_pdb(cif_path, ligand_chain, template, staged_path)
    log(f"[{complex_id}] staged pose -> {staged_path}")

    try:
        replicas = rl.relieve_clashes(
            staged_path, config.all_params_paths(), ligand_resname,
            n_replicas=n_replicas, relax_cycles=relax_cycles, seed=seed,
        )

        frames = []
        for rep in replicas:
            log(f"[{complex_id}] replica {rep['replica']}: "
                f"fa_rep {rep['fa_rep_raw']:.1f} -> {rep['fa_rep_relaxed']:.1f}  "
                f"total {rep['total_raw']:.1f} -> {rep['total_relaxed']:.1f}")
            sfxn = pyrosetta.get_score_function()
            df = dc.decompose_ligand_contacts(rep["pose"], sfxn, ligand_resname)
            df.insert(0, "replica", rep["replica"])
            df.insert(0, "ligand_pose_cluster", ligand_pose_cluster)
            df.insert(0, "ca_cluster", ca_cluster)
            df.insert(0, "ligand", ligand)
            df.insert(0, "protein", protein)
            df.insert(0, "complex_id", complex_id)
            df["fa_rep_raw"] = rep["fa_rep_raw"]
            df["fa_rep_relaxed"] = rep["fa_rep_relaxed"]
            df["total_raw"] = rep["total_raw"]
            df["total_relaxed"] = rep["total_relaxed"]
            frames.append(df)
    finally:
        # staged_path is pure intermediate scratch (regenerable from cif_path any
        # time via pose_prep.prepare_complex_pdb) -- deleting it regardless of
        # success/failure keeps disk usage flat across a batch of thousands of
        # complexes instead of growing by ~350 KB/complex forever. The per-complex
        # .log already captures enough detail to debug a failure without the PDB.
        staged_path.unlink(missing_ok=True)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log(f"[{complex_id}] done in {time.time() - t0:.1f}s, {len(result)} rows")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex-id", required=True)
    ap.add_argument("--n-replicas", type=int, default=1)
    ap.add_argument("--relax-cycles", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                     help="fixed Rosetta RNG seed for a fully reproducible run (omit for genuine ensemble variation)")
    args = ap.parse_args()

    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows = manifest[manifest["complex_id"] == args.complex_id]
    if rows.empty:
        sys.exit(f"complex_id {args.complex_id!r} not found in {config.MANIFEST_CSV}")
    row = rows.iloc[0]

    ligand_chain, smiles = pp.resolved_ligand_chain(config.ABCFOLD_OUT_ROOT / f"{row['protein']}__holo")
    template = lf.build_template(smiles, config.ligand_resname(row["ligand"]))

    df = run_one(
        row["complex_id"], config.PIPELINE_ROOT / row["cif_path"], row["protein"], row["ligand"],
        ligand_chain, template, row["ca_cluster"], row["ligand_pose_cluster"],
        n_replicas=args.n_replicas, relax_cycles=args.relax_cycles, seed=args.seed,
    )
    out_path = config.PER_COMPLEX_DIR / f"{row['complex_id']}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
