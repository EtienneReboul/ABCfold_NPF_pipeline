#!/usr/bin/env python3
"""
rescoring/src/rescore_redocked_batch.py
===========================================
Score every ../../redocking/ HADDOCK3-redocked complex's top-ranked model
with PyRosetta -- WITHOUT relief.py's FastRelax. HADDOCK3's own [flexref]
step already refined the pose (semi-flexible CNS minimization); stacking
Rosetta's own coordinate-constrained FastRelax on top of that would be a
second, inconsistent relaxation under a different force field, not a
meaningful "clash relief" step the way it is for ABCfold's raw, never-
minimized poses (see rescoring/README.md's "Raw (non-preminimized) poses"
section -- that whole rationale for relief.py doesn't apply here). Per the
user (2026-08-26): score the HADDOCK3 pose exactly as HADDOCK3 produced it.

Deliberately self-contained in rescoring/src/ (imports only this project's
own config/pose_prep/ligand_fix/decompose, plus pyrosetta) rather than
cross-importing redocking/src/config.py or redocking/src/compare_to_abcfold.py
-- both this project and redocking/ have their OWN module named `config`,
and Python's import cache is keyed by module name: whichever one loads
first would silently shadow the other for every subsequent bare `import
config` in either package's own modules (pose_prep.py/ligand_fix.py both do
`import config` internally, expecting THIS project's config). Keeping every
cross-project reference here to plain file paths (never importing
redocking's own modules) sidesteps that entirely -- same reasoning
run_chimerax_try.py's module docstring gives for splitting across
subprocesses for a *different* kind of environment isolation.

Reuses pose_prep.py/ligand_fix.py/decompose.py completely unchanged:
HADDOCK3's ligand (chain "B", from redocking/src/make_haddock_cfg.py's
molecules=[receptor, ligand] ordering, confirmed correct against a real
run's own log) is built from the exact same GA1 SMILES
(config.yaml -> config.load_ligand_smiles("GA1")) via RDKit's
ConstrainedEmbed in redocking/src/build_ga1_from_ga3.py, so the same
positional (not name-based) heavy-atom correspondence ligand_fix.py already
relies on for all 6 ABCfold backends holds here too -- just point
pose_prep.prepare_complex_pdb at a HADDOCK3 model.pdb instead of an ABCfold
CIF (gemmi reads PDB/mmCIF/gzipped-either alike, confirmed by hand on a
real flexref_*.pdb.gz).

Output: redocking/results/rescoring/per_complex/<complex_id>.csv, same
column shape as rescoring/results/per_complex/*.csv (this project's own
ab-initio-pose scoring) plus `role`/`ca_cluster`/`form` from redocking's
manifest, so aggregate scripts can pool both without reshaping. Resumable
-- skips any complex_id that already has an output CSV.

Usage:
    python rescore_redocked_batch.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import pyrosetta
from pyrosetta import rosetta

import config
import decompose as dc
import ligand_fix as lf
import pose_prep as pp

LIGAND_KEY = "GA1"
LIGAND_CHAIN_HADDOCK = "B"  # confirmed correct against a real HADDOCK3 run's own log (see redocking/src/compare_to_abcfold.py)

REDOCKING_ROOT = config.PIPELINE_ROOT / "redocking"
REDOCKING_MANIFEST_CSV = REDOCKING_ROOT / "data" / "manifest.csv"
REDOCKING_HADDOCK_RUNS_DIR = REDOCKING_ROOT / "results" / "haddock_runs"

OUT_DIR = REDOCKING_ROOT / "results" / "rescoring" / "per_complex"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_final_caprieval_dir(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*_caprieval"), key=lambda p: int(p.name.split("_")[0]))
    return candidates[-1] if candidates else None


def top_ranked_model_path(run_dir: Path) -> tuple[Path, dict] | None:
    """Best-HADDOCK-score model from this complex's final caprieval step --
    None if the run hasn't reached (or never will reach) that step. Handles
    HADDOCK3's in-place gzip of kept models (see module docstring)."""
    caprieval_dir = _find_final_caprieval_dir(run_dir)
    if caprieval_dir is None:
        return None
    tsv_path = caprieval_dir / "capri_ss.tsv"
    if not tsv_path.exists():
        return None
    with tsv_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r["score"]))
    model_path = Path(best["model"])
    if not model_path.is_absolute():
        model_path = caprieval_dir / model_path
    if not model_path.exists():
        gz_path = model_path.with_suffix(model_path.suffix + ".gz")
        if gz_path.exists():
            model_path = gz_path
    return model_path, best


def score_one(complex_id: str, model_path: Path, capri_row: dict, manifest_row: dict) -> pd.DataFrame:
    ligand_resname = config.ligand_resname(LIGAND_KEY)
    smiles = config.load_ligand_smiles(LIGAND_KEY)
    template = lf.build_template(smiles, ligand_resname)

    staged_path = config.STAGED_DIR / f"redock_{complex_id}.pdb"
    pp.prepare_complex_pdb(model_path, LIGAND_CHAIN_HADDOCK, template, staged_path)

    try:
        pose = pyrosetta.pose_from_pdb(str(staged_path))
        sfxn = pyrosetta.get_score_function()
        sfxn(pose)
        fa_rep = pose.energies().total_energies()[rosetta.core.scoring.fa_rep]
        total_score = pose.energies().total_energies()[rosetta.core.scoring.total_score]
        df = dc.decompose_ligand_contacts(pose, sfxn, ligand_resname)
    finally:
        # pure scratch, same as rescoring/src/run_complex.py's own staged_path -- regenerable
        # from model_path any time, never trust anything but decompose's own returned df.
        staged_path.unlink(missing_ok=True)

    df.insert(0, "complex_id", complex_id)
    df["protein"] = manifest_row["protein"]
    df["role"] = manifest_row["role"]
    df["form"] = manifest_row["form"]
    df["ca_cluster"] = manifest_row["ca_cluster"]
    df["haddock_score"] = float(capri_row["score"])
    df["fa_rep_haddock_pose"] = fa_rep
    df["total_score_haddock_pose"] = total_score
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N manifest rows (smoke-testing)")
    args = ap.parse_args()

    if not REDOCKING_MANIFEST_CSV.exists():
        sys.exit(f"{REDOCKING_MANIFEST_CSV} not found -- run redocking/src/make_manifest.py first.")
    with REDOCKING_MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    pyrosetta.init(f"-extra_res_fa {config.params_path(LIGAND_KEY)} -mute all")

    n_scored, n_skipped_done, n_skipped_no_output, n_failed = 0, 0, 0, 0
    for row in rows:
        complex_id = row["complex_id"]
        out_path = OUT_DIR / f"{complex_id}.csv"
        if out_path.exists():
            n_skipped_done += 1
            continue

        run_dir = REDOCKING_HADDOCK_RUNS_DIR / complex_id
        found = top_ranked_model_path(run_dir)
        if found is None:
            print(f"[rescore_redocked] {complex_id}: no completed caprieval output yet -- skipping")
            n_skipped_no_output += 1
            continue
        model_path, capri_row = found

        t0 = time.time()
        try:
            df = score_one(complex_id, model_path, capri_row, row)
        except Exception as exc:
            print(f"[rescore_redocked] {complex_id}: FAILED -- {exc}")
            n_failed += 1
            continue
        df.to_csv(out_path, index=False)
        n_scored += 1
        print(f"[rescore_redocked] {complex_id}: {len(df)} contact rows, "
              f"haddock_score={capri_row['score']}, done in {time.time() - t0:.1f}s -> {out_path}")

    print(f"[rescore_redocked] done: {n_scored} scored, {n_skipped_done} already had output, "
          f"{n_skipped_no_output} not yet completed on the cluster, {n_failed} failed")


if __name__ == "__main__":
    main()
