"""
redocking/src/run_haddock_batch.py
=====================================
Stage 6: submit each manifest complex's HADDOCK3 run
(results/haddock_runs/<complex_id>/run.cfg, from make_haddock_cfg.py) as
its own SLURM job on the IFB cluster -- same cluster ABCfold's own cofolding
runs on (worflows/processing/submit_abcfold.sh), but far lighter-weight:
HADDOCK3/CNS is CPU-only (no GPU/Docker/Singularity needed), and this
pilot is 3 complexes, not a large array, so one `sbatch` per complex_id is
enough -- revisit as a proper job array (submit_abcfold.sh's --batch-size
chunking pattern) only if/when this pilot scales past a handful of
complexes.

**Partition/account/time defaults below are PLACEHOLDERS**, not confirmed
against IFB's actual CPU-partition names the way submit_abcfold.sh's
`PARTITION="gpu"` was confirmed for GPU jobs (see that script's own
"CONFIRMED on IFB" comment) -- check `sinfo` on the cluster and override
via --partition/--account before the first real submission.

Requires: haddock3 installed in a conda env on the cluster
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

DEFAULT_PARTITION = "fast"  # PLACEHOLDER -- confirm the real CPU partition name via `sinfo` on IFB
DEFAULT_CPUS = 8            # HADDOCK3 parallelizes across models within a run via multiprocessing
DEFAULT_MEM = "16G"
DEFAULT_TIME = "08:00:00"
DEFAULT_ENV_NAME = "redocking"

SLURM_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{account_line}
#SBATCH --job-name=redock_{complex_id}
#SBATCH --output={log_dir}/{complex_id}.log
#SBATCH --error={log_dir}/{complex_id}.err
set -euo pipefail

HADDOCK3="{haddock3_bin}"
echo "[$(date)] redocking {complex_id}"
echo "  haddock3: $HADDOCK3"
"$HADDOCK3" "{run_cfg}"
echo "[$(date)] done {complex_id}"
"""


def submit_complex(complex_id: str, run_cfg: Path, log_dir: Path, haddock3_bin: str,
                    partition: str, cpus: int, mem: str, time: str, account: str, dry_run: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    account_line = f"#SBATCH --account={account}" if account else ""
    script = SLURM_SCRIPT_TEMPLATE.format(
        partition=partition, cpus=cpus, mem=mem, time=time, account_line=account_line,
        complex_id=complex_id, log_dir=log_dir, haddock3_bin=haddock3_bin, run_cfg=run_cfg,
    )
    script_path = run_cfg.parent / "submit.sh"
    script_path.write_text(script)

    if dry_run:
        print(f"[dry-run] would submit {script_path}")
        return
    subprocess.run(["sbatch", str(script_path)], check=True)
    print(f"submitted {complex_id} ({script_path})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--mem", default=DEFAULT_MEM)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--account", default="")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME,
                         help="Conda env name haddock3 is installed in (resolved to an absolute "
                              "bin/haddock3 path, IFB's ~/.condarc envs_dirs convention -- see "
                              "submit_abcfold.sh's METADATA_PYTHON comment for why absolute path, "
                              "not `conda activate`, inside a job script).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    haddock3_bin = f"/shared/projects/npf_abinitio/conda/envs/{args.env_name}/bin/haddock3"
    log_dir = config.HADDOCK_RUNS_DIR / "_slurm_logs"

    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        complex_id = row["complex_id"]
        run_cfg = config.HADDOCK_RUNS_DIR / complex_id / "run.cfg"
        if not run_cfg.exists():
            raise FileNotFoundError(f"{run_cfg} not found -- run make_haddock_cfg.py first.")
        submit_complex(complex_id, run_cfg, log_dir, haddock3_bin,
                        args.partition, args.cpus, args.mem, args.time, args.account, args.dry_run)


if __name__ == "__main__":
    main()
