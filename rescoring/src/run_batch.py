#!/usr/bin/env python3
"""
rescoring/src/run_batch.py
=============================
Batch the whole manifest (or a --protein-restricted subset — this is what
worflows/postprocessing/Snakefile's rescoring_run_complex rule calls, one
Snakemake job per protein) with a single command, per-complex logs,
resumable (skips complexes that already have a
results/per_complex/<complex_id>.csv — safe to Ctrl-C and re-run).

Uses one OS process per worker (multiprocessing, not threads) — PyRosetta is
not thread-safe within a process, but each worker process calls
pyrosetta.init() exactly once (relief.py guards it) and is otherwise
independent.

Usage:
    python run_batch.py [--workers N] [--n-replicas 1] [--relax-cycles 1]
                         [--protein NPF2.12_Q9LFX9] [--ligand GA1] [--limit 50]
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib

import pandas as pd

import config
import ligand_fix as lf
import pose_prep as pp
import run_complex as rc


def _complex_seed(base_seed: int, complex_id: str) -> int:
    """Per-complex seed derived from (base_seed, complex_id) — NOT the same
    base_seed reused for every complex. Work is distributed across worker
    processes by imap_unordered in whatever order the OS scheduler picks, so
    a shared global seed would make each complex's result depend on which
    worker happened to process it first. Hashing the complex_id in makes
    every complex's seed independent of scheduling."""
    return (base_seed + zlib.crc32(complex_id.encode())) % (2**31 - 1)


_TEMPLATE_CACHE: dict[str, object] = {}
_LIGAND_CHAIN_CACHE: dict[str, str] = {}


def _template_for(ligand: str):
    if ligand not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[ligand] = lf.build_template(config.load_ligand_smiles(ligand), config.ligand_resname(ligand))
    return _TEMPLATE_CACHE[ligand]


def _ligand_chain_for(protein: str) -> str:
    if protein not in _LIGAND_CHAIN_CACHE:
        chain, _smiles = pp.resolved_ligand_chain(config.ABCFOLD_OUT_ROOT / f"{protein}__holo")
        _LIGAND_CHAIN_CACHE[protein] = chain
    return _LIGAND_CHAIN_CACHE[protein]


def _worker(args):
    (complex_id, cif_rel, protein, ligand, ca_cluster, ligand_pose_cluster,
     n_replicas, relax_cycles, base_seed) = args
    seed = _complex_seed(base_seed, complex_id) if base_seed is not None else None
    log_path = config.LOGS_DIR / f"{complex_id}.log"
    out_path = config.PER_COMPLEX_DIR / f"{complex_id}.csv"
    try:
        ligand_chain = _ligand_chain_for(protein)
        template = _template_for(ligand)
        with open(log_path, "w") as log_file:
            df = rc.run_one(
                complex_id, config.PIPELINE_ROOT / cif_rel, protein, ligand, ligand_chain, template,
                ca_cluster, ligand_pose_cluster,
                n_replicas=n_replicas, relax_cycles=relax_cycles,
                seed=seed, log_file=log_file,
            )
        df.to_csv(out_path, index=False)
        return complex_id, True, None
    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"FAILED: {e}\n{traceback.format_exc()}")
        return complex_id, False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--n-replicas", type=int, default=1)
    ap.add_argument("--relax-cycles", type=int, default=1)
    ap.add_argument("--protein", action="append", help="restrict to this protein (repeatable)")
    ap.add_argument("--ligand", action="append", help="restrict to this ligand key (repeatable)")
    ap.add_argument("--limit", type=int, help="only process the first N (post-filter, post-resume) complexes")
    ap.add_argument("--seed", type=int, default=None,
                     help="fixed base seed for a fully reproducible batch (per-complex seeds are derived "
                          "from this + complex_id, independent of worker scheduling); omit for genuine "
                          "ensemble variation")
    args = ap.parse_args()

    missing_params = set()
    manifest = pd.read_csv(config.MANIFEST_CSV)
    if args.protein:
        manifest = manifest[manifest["protein"].isin(args.protein)]
    if args.ligand:
        manifest = manifest[manifest["ligand"].isin(args.ligand)]
    for ligand in manifest["ligand"].unique():
        if not config.params_path(ligand).exists():
            missing_params.add(ligand)
    if missing_params:
        sys.exit(f"Missing params for ligand(s) {sorted(missing_params)} -- run prep_ligand.py first.")

    done = {p.stem for p in config.PER_COMPLEX_DIR.glob("*.csv")}
    todo = manifest[~manifest["complex_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"[run_batch] {len(manifest)} in manifest, {len(done)} already done, "
          f"{len(todo)} to run now, {args.workers} workers")
    if todo.empty:
        return

    tasks = [
        (row["complex_id"], row["cif_path"], row["protein"], row["ligand"],
         row["ca_cluster"], row["ligand_pose_cluster"],
         args.n_replicas, args.relax_cycles, args.seed)
        for _, row in todo.iterrows()
    ]

    t0 = time.time()
    n_ok, n_fail = 0, 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (complex_id, ok, err) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            n_ok += ok
            n_fail += not ok
            status = "OK" if ok else f"FAILED ({err})"
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_min = (len(tasks) - i) / rate / 60 if rate > 0 else float("nan")
            print(f"[run_batch] {i}/{len(tasks)} {complex_id}: {status}  "
                  f"({rate:.2f}/s, ETA {eta_min:.0f} min)")

    print(f"[run_batch] done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
