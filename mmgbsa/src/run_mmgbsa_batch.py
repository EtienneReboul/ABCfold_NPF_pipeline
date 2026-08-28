"""
mmgbsa/src/run_mmgbsa_batch.py
==============================
Stage 4: one SLURM job array on the `fast` (CPU) partition that runs
`gmx_MMPBSA MPI` for every (complex, replica) whose Stage 3 production
trajectory exists -- GB (igb=8) endpoint free energy + per-residue
decomposition.

mmpbsa.in (written per rep dir here, not at runtime):
  &general  startframe / endframe / interval, temperature=300
  &gb       igb=8, saltcon=0.150
  &decomp   idecomp=2 (per-residue, sidechain/backbone split),
            dec_verbose=3, print_res="within 5", csv_format=1

Inputs per task:
  -cs prod.tpr -ci <sys>/index.ndx -cg Protein GA1 -ct prod.xtc
  -cp <sys>/topol.top -lm data/ligand_params/GA1.mol2

Outputs copied to results/mmgbsa/<complex_id>/rep<k>/:
  FINAL_RESULTS_MMPBSA.dat   total dG_GB (+ components)
  FINAL_DECOMP_MMPBSA.csv    per-residue delta decomposition  <-- Stage 5 parses this

Same array-manifest / absolute-env-path / resumable pattern as
run_md_batch.py. CPU/MPI (`fast` partition), no GPU. `gmx_MMPBSA` shells out
to `gmx` (trajectory conversion) and to this env's AmberTools, so the job
script `module load`s gromacs first and falls back to the conda `gmx`, same
as run_md_batch.py. Resumable: a rep whose FINAL_DECOMP_MMPBSA.csv already
exists is skipped.

Defaults target **pangloss** ($HOME/miniforge3/envs, `fast` partition). For
IFB pass `--conda-root /shared/projects/npf_abinitio/conda/envs`.

Usage:
    python run_mmgbsa_batch.py [--dry-run] [--limit N] [--smoke]
        [--np 8] [--max-concurrent K] [--conda-root DIR] [--gromacs-module NAME]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config

DEFAULT_CONDA_ROOT = "$HOME/miniforge3/envs"
DEFAULT_ENV = "mmgbsa"
DEFAULT_PARTITION = "fast"
DEFAULT_NP = 8
DEFAULT_MEM = "24G"
DEFAULT_TIME = "06:00:00"
DEFAULT_MAX_CONCURRENT = 12
DEFAULT_GROMACS_MODULE = "gromacs"

PROD_ENDFRAME = 500      # 5 ns / 10 ps
SMOKE_ENDFRAME = 10      # 100 ps / 10 ps


def mmpbsa_in(endframe: int, interval: int) -> str:
    return f"""\
Per-residue GB decomposition of a HADDOCK3-redocked GA1 pose (mmgbsa/ pilot)
&general
  sys_name           = "GA1_MMGBSA"
  startframe          = 1
  endframe            = {endframe}
  interval           = {interval}
  temperature        = 300
  verbose            = 2
/
&gb
  igb                = {config.IGB}
  saltcon            = {config.SALT_MOLAR}
/
&decomp
  idecomp            = 2
  dec_verbose        = 3
  print_res          = "within 5"
  csv_format         = 1
/
"""


JOB_SCRIPT = """\
#!/usr/bin/env bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={np}
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last}%{maxc}
#SBATCH --job-name=mmgbsa_gb
#SBATCH --output={logdir}/gb_%a.log
#SBATCH --error={logdir}/gb_%a.err
set -euo pipefail

ENV="{conda_root}/{env}"
export PATH="$ENV/bin:$PATH"
GMX_BIN="{gmx_bin}"
if [[ -n "$GMX_BIN" ]]; then
    export PATH="$(dirname "$GMX_BIN"):$PATH"
    [[ -d "$ENV/share/gromacs/top" ]] && export GMXLIB="$ENV/share/gromacs/top"
else
    module load {gromacs_module} 2>/dev/null || true   # conda gmx ($ENV/bin) is the fallback
fi
echo "[gb] gmx: $(command -v gmx || echo NONE)"
gmx --version 2>/dev/null | grep -E "GROMACS version" || true
MANIFEST="{manifest}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[[ -z "$LINE" ]] && {{ echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 0; }}
CID=$(echo "$LINE" | cut -f1)
REP=$(echo "$LINE" | cut -f2)

SYS="{systems_dir}/$CID"
MD="{md_dir}/$CID/rep$REP"
OUT="{mmgbsa_dir}/$CID/rep$REP"
mkdir -p "$OUT"

if [[ -f "$OUT/FINAL_DECOMP_MMPBSA.csv" ]]; then
    echo "[gb] $CID rep$REP already done -- skipping"; exit 0
fi
if [[ ! -f "$MD/prod.xtc" || ! -f "$MD/prod.tpr" ]]; then
    echo "[gb] $CID rep$REP: no production trajectory yet -- skipping"; exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp "$OUT/mmpbsa.in" "$WORK/"
cd "$WORK"

echo "[gb] $(date) $CID rep$REP"
mpirun -np {np} gmx_MMPBSA MPI -O \\
    -i mmpbsa.in \\
    -cs "$MD/prod.tpr" \\
    -ci "$SYS/index.ndx" \\
    -cg Protein GA1 \\
    -ct "$MD/prod.xtc" \\
    -cp "$SYS/topol.top" \\
    -lm "{ligand_mol2}" \\
    -o FINAL_RESULTS_MMPBSA.dat \\
    -do FINAL_DECOMP_MMPBSA.dat \\
    -eo FINAL_RESULTS_MMPBSA.csv \\
    -deo FINAL_DECOMP_MMPBSA.csv \\
    -nogui

cp FINAL_RESULTS_MMPBSA.dat FINAL_RESULTS_MMPBSA.csv FINAL_DECOMP_MMPBSA.dat FINAL_DECOMP_MMPBSA.csv "$OUT/" 2>/dev/null || true
echo "[gb] $(date) done $CID rep$REP"
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--np", type=int, default=DEFAULT_NP)
    ap.add_argument("--partition", default=DEFAULT_PARTITION)
    ap.add_argument("--time", default=DEFAULT_TIME)
    ap.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    ap.add_argument("--conda-root", default=DEFAULT_CONDA_ROOT)
    ap.add_argument("--env", default=DEFAULT_ENV)
    ap.add_argument("--gromacs-module", default=DEFAULT_GROMACS_MODULE,
                    help="site module to `module load` for gmx; '' or 'none' to skip and use the conda gmx")
    ap.add_argument("--gmx-bin", default="",
                    help="explicit gmx binary (its dir is prepended to PATH; needed if the Stage 3 "
                         "trajectory/tpr was written by a GROMACS newer than the conda one). '' = conda/module.")
    args = ap.parse_args()

    rows = config.read_csv_rows(config.MANIFEST_CSV)
    if args.smoke:
        rows = config.smoke_rows(rows)
    if args.limit:
        rows = rows[: args.limit]

    endframe = SMOKE_ENDFRAME if args.smoke else PROD_ENDFRAME
    interval = 1 if args.smoke else config.GB_FRAME_INTERVAL

    tasks: list[tuple[str, int]] = []
    for r in rows:
        for rep in range(config.N_REPLICAS):
            rep_dir = config.MMGBSA_DIR / r["complex_id"] / f"rep{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            (rep_dir / "mmpbsa.in").write_text(mmpbsa_in(endframe, interval))
            tasks.append((r["complex_id"], rep))
    if not tasks:
        print("[stage4] no tasks"); return

    manifest = config.SLURM_CFG_DIR / ("gb_array_manifest.smoke.txt" if args.smoke else "gb_array_manifest.txt")
    manifest.write_text("\n".join(f"{c}\t{rep}" for c, rep in tasks) + "\n")

    gmod = "true" if args.gromacs_module.lower() in ("", "none") else args.gromacs_module
    gmx_bin = "" if args.gmx_bin.lower() in ("", "none") else args.gmx_bin
    script = JOB_SCRIPT.format(
        partition=args.partition, np=args.np, mem=DEFAULT_MEM, time=args.time,
        last=len(tasks) - 1, maxc=args.max_concurrent, logdir=config.SLURM_LOG_DIR,
        manifest=manifest, systems_dir=config.SYSTEMS_DIR, md_dir=config.MD_DIR,
        mmgbsa_dir=config.MMGBSA_DIR, conda_root=args.conda_root, env=args.env,
        gromacs_module=gmod, gmx_bin=gmx_bin, ligand_mol2=config.LIGAND_PARAMS_DIR / "GA1.mol2",
    )
    script_path = config.SLURM_CFG_DIR / ("submit_gb.smoke.sh" if args.smoke else "submit_gb.sh")
    script_path.write_text(script)
    print(f"[stage4] {len(tasks)} tasks (endframe={endframe}, interval={interval}) -> {manifest}")
    print(f"[stage4] array script: {script_path}")

    if args.dry_run:
        print(f"[stage4] --dry-run: would `sbatch {script_path}`")
        return
    import subprocess
    subprocess.run(["sbatch", str(script_path)], check=True)
    print("[stage4] submitted")


if __name__ == "__main__":
    main()
