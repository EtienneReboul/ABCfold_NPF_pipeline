"""
redocking/src/run_haddock_batch.py
=====================================
Stage 6: submit every manifest complex's HADDOCK3 run
(results/haddock_runs/_cfgs/<complex_id>.cfg, from make_haddock_cfg.py) as
ONE SLURM job array on the IFB cluster -- same cluster ABCfold's own
cofolding runs on, and the same array + manifest-file pattern
worflows/processing/submit_abcfold.sh already uses (one line per task,
`SLURM_ARRAY_TASK_ID` indexes into it inside the job script) -- reused
here rather than one `sbatch` call per complex, since this pipeline is
scaling from a 3-complex pilot to a ~24+ complex batch and a plain loop of
individual submissions gets unwieldy (and harder to monitor/cancel as one
unit) at that size. Each array task = exactly one HADDOCK3 run (no
per-task batching the way submit_abcfold.sh's BATCH_SIZE does -- a single
HADDOCK3 run is already a substantial, non-trivial unit of work, unlike
one ABCfold prediction).

HADDOCK3/CNS is CPU-only (no GPU/Docker/Singularity needed) -- far
lighter-weight than the ABCfold array this mirrors.

**Partition confirmed via `sinfo` on IFB (2026-08-25): "fast"** (default
below). Requires haddock3 installed in a conda env on the cluster
(envs/redocking.yaml), resolved by absolute env path (not `conda
activate`), same reasoning submit_abcfold.sh's own inline comment gives
for avoiding conda activation inside a job script.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import config

DEFAULT_PARTITION = "fast"
DEFAULT_CPUS = 32
DEFAULT_MEM = "128G"
DEFAULT_TIME = "08:00:00"
DEFAULT_ENV_NAME = "redocking"
DEFAULT_MAX_CONCURRENT = 8  # array tasks running at once -- 8*32=256 cores, a considerate
                            # chunk of the shared "fast" partition rather than the whole thing
                            # at once; raise with --max-concurrent if the queue allows more.

ARRAY_JOB_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{account_line}
#SBATCH --array=0-{last_task}%{max_concurrent}
#SBATCH --job-name=redock_array
#SBATCH --output={log_dir}/task_%a.log
#SBATCH --error={log_dir}/task_%a.err
set -euo pipefail

HADDOCK3="{haddock3_bin}"
MANIFEST="{manifest}"
TASK_ID=$SLURM_ARRAY_TASK_ID
LINE=$(( TASK_ID + 1 ))

CFG=$(sed -n "${{LINE}}p" "$MANIFEST")
if [[ -z "$CFG" ]]; then
    echo "[$(date)] task $TASK_ID: no manifest line $LINE -- nothing to do"
    exit 0
fi
COMPLEX_ID=$(basename "$CFG" .cfg)

echo "[$(date)] task $TASK_ID: redocking $COMPLEX_ID"
echo "  cfg: $CFG"
echo "  haddock3: $HADDOCK3"
"$HADDOCK3" "$CFG"
echo "[$(date)] task $TASK_ID: done $COMPLEX_ID"
"""


def write_manifest(cfg_paths: list[Path], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(str(p) for p in cfg_paths) + "\n")
    return manifest_path


def submit_array(cfg_paths: list[Path], manifest_path: Path, log_dir: Path, haddock3_bin: str,
                  partition: str, cpus: int, mem: str, time: str, account: str,
                  max_concurrent: int, dry_run: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    account_line = f"#SBATCH --account={account}" if account else ""
    script = ARRAY_JOB_SCRIPT_TEMPLATE.format(
        partition=partition, cpus=cpus, mem=mem, time=time, account_line=account_line,
        last_task=len(cfg_paths) - 1, max_concurrent=max_concurrent, log_dir=log_dir,
        haddock3_bin=haddock3_bin, manifest=manifest_path,
    )
    script_path = manifest_path.parent / "submit_array.sh"
    script_path.write_text(script)

    print(f"{len(cfg_paths)} complexes -> {manifest_path}")
    print(f"Array script: {script_path} (--array=0-{len(cfg_paths) - 1}%{max_concurrent})")

    if dry_run:
        print(f"[dry-run] would submit {script_path}")
        return
    subprocess.run(["sbatch", str(script_path)], check=True)
    print("submitted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--mem", default=DEFAULT_MEM)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--account", default="")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                         help="Max array tasks running at once (SLURM's --array=0-N%%K throttle).")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME,
                         help="Conda env name haddock3 is installed in (resolved to an absolute "
                              "bin/haddock3 path, IFB's ~/.condarc envs_dirs convention -- see "
                              "submit_abcfold.sh's METADATA_PYTHON comment for why absolute path, "
                              "not `conda activate`, inside a job script).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                         help="skip any complex whose results/haddock_runs/<complex_id>/ run_dir already "
                              "exists and is non-empty -- HADDOCK3 itself refuses to write into a non-empty "
                              "run_dir, so resubmitting an already-completed complex would just fail "
                              "immediately and waste an array slot. Use this when extending an existing "
                              "manifest with new rows (e.g. after make_manifest.py adds new proteins) rather "
                              "than starting a run from scratch.")
    args = parser.parse_args()

    haddock3_bin = f"/shared/projects/npf_abinitio/conda/envs/{args.env_name}/bin/haddock3"
    cfgs_dir = config.HADDOCK_RUNS_DIR / "_cfgs"
    log_dir = config.HADDOCK_RUNS_DIR / "_slurm_logs"

    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))

    cfg_paths = []
    n_skipped = 0
    for row in rows:
        if args.skip_existing:
            run_dir = config.HADDOCK_RUNS_DIR / row["complex_id"]
            if run_dir.exists() and any(run_dir.iterdir()):
                n_skipped += 1
                continue
        cfg_path = cfgs_dir / f"{row['complex_id']}.cfg"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found -- run make_haddock_cfg.py first.")
        cfg_paths.append(cfg_path)

    if args.skip_existing:
        print(f"--skip-existing: {n_skipped} complex(es) already have a run_dir, submitting the "
              f"remaining {len(cfg_paths)}/{len(rows)}")

    manifest_path = cfgs_dir / "array_manifest.txt"
    write_manifest(cfg_paths, manifest_path)
    submit_array(cfg_paths, manifest_path, log_dir, haddock3_bin,
                 args.partition, args.cpus, args.mem, args.time, args.account,
                 args.max_concurrent, args.dry_run)


if __name__ == "__main__":
    main()
