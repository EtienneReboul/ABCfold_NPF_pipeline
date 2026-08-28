"""
mmgbsa/src/run_md_batch.py
==========================
Stage 3: one SLURM job array that runs the short MD for every
(complex, replica) pair -- 143 complexes x 3 replicas = 429 tasks. Same
array + array-manifest-file pattern as redocking/src/run_haddock_batch.py
and worflows/processing/submit_abcfold.sh (one line per task,
$SLURM_ARRAY_TASK_ID indexes it), same "resolve the env by absolute path,
never `conda activate` inside a job script" convention.

Each task, for its one (complex, replica):
  grompp+mdrun  em -> nvt -> npt -> prod
into results/md/<complex_id>/rep<k>/. prod.mdp is written per task with the
replica's own ld_seed. Resumable: a task whose prod.gro already exists exits
immediately; a partial prod is continued with `mdrun -cpi prod.cpt`.

GROMACS resolution (checked 2026-08-27: neither IFB nor pangloss has a
gromacs module by default): the job script does `module load gromacs` and
uses that `gmx` if it appears; otherwise it falls back to the conda env's
own `bin/gmx` (the portable CPU build in envs/mmgbsa.yaml). Pass a specific
module name with --gromacs-module.

Defaults target **pangloss** (IBMP): `cryoem` partition, `gpu:l4:1`, conda
env under $HOME/miniforge3/envs. For IFB use
`--conda-root /shared/projects/npf_abinitio/conda/envs --partition gpu
--gres gpu:l40s:1 --gromacs-module '' ` (IFB has no gromacs module, so the
conda CPU build is the only option there unless one gets installed).

Usage:
    python run_md_batch.py [--dry-run] [--smoke] [--cpu]
        [--partition P] [--gres G] [--conda-root DIR] [--gromacs-module NAME]
        [--max-concurrent K] [--time HH:MM:SS]
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import config

# pangloss defaults. GROMACS = the CUDA build installed into the conda env
# itself (`CONDA_OVERRIDE_CUDA=12.4 mamba install -n mmgbsa "gromacs=*=nompi_cuda_*"`)
# -- self-contained, version-consistent across every stage, and has the PTX
# fallback so it runs on niobe's V100S (SM 7.0). dpflieger's hand-built
# /shared/home/dpflieger/Shared/gmx only targets SM>=7.5, so it silently fell
# back to CPU on niobe; pass --gmx-bin to use it (or a module `gmx`) instead.
DEFAULT_CONDA_ROOT = "$HOME/miniforge3/envs"
DEFAULT_ENV = "mmgbsa"
DEFAULT_PARTITION = "gpu"             # node niobe, 2x Tesla V100S (IT: "toujours via niobe")
DEFAULT_GRES = "gpu:tesla:1"
DEFAULT_CPUS = 16
DEFAULT_MEM = "32G"
DEFAULT_TIME = "24:00:00"
DEFAULT_MAX_CONCURRENT = 2            # 2 GPUs on niobe; also the per-user cap on IFB
DEFAULT_GROMACS_MODULE = "gromacs"    # used only if --gmx-bin is not given and no conda gmx
DEFAULT_GMX_BIN = ""                  # "" -> $ENV/bin/gmx (the conda CUDA build)
CPU_PARTITION = "fast"
SMOKE_PROD_NSTEPS = 50000             # 100 ps

JOB_SCRIPT = """\
#!/usr/bin/env bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{gres_line}
#SBATCH --time={time}
#SBATCH --array=0-{last}%{maxc}
#SBATCH --job-name=mmgbsa_md
#SBATCH --output={logdir}/md_%a.log
#SBATCH --error={logdir}/md_%a.err
set -euo pipefail

ENV="{conda_root}/{env}"
PY="$ENV/bin/python"
MANIFEST="{manifest}"
SRC_DIR="{src_dir}"
PROD_NSTEPS="{prod_nsteps}"
USE_GPU="{use_gpu}"
GMX_BIN="{gmx_bin}"

# --- resolve gmx: explicit --gmx-bin > site module > conda env fallback ---
if [[ -n "$GMX_BIN" ]]; then
    GMX="$GMX_BIN"
    # dpflieger's build tree has no share/gromacs/top -- point GMXLIB at the
    # conda env's complete force-field data (amber99sb-ildn / tip3p / ions are
    # format-stable across 2024<->2026), so grompp/pdb2gmx can resolve includes.
    [[ -d "$ENV/share/gromacs/top" ]] && export GMXLIB="$ENV/share/gromacs/top"
else
    module load {gromacs_module} 2>/dev/null || true
    GMX="$(command -v gmx || true)"
    [[ -z "$GMX" ]] && GMX="$ENV/bin/gmx"
fi
export PATH="$(dirname "$GMX"):$PATH"
echo "[md] using GMX=$GMX  GMXLIB=${{GMXLIB:-<default>}}"
"$GMX" --version | grep -E "GROMACS version|GPU support" || true

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[[ -z "$LINE" ]] && {{ echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 0; }}
CID=$(echo "$LINE" | cut -f1)
REP=$(echo "$LINE" | cut -f2)
SEED=$(echo "$LINE" | cut -f3)

SYS="{systems_dir}/$CID"
OUT="{md_dir}/$CID/rep$REP"
mkdir -p "$OUT"
cd "$OUT"

if [[ -f prod.gro ]]; then
    echo "[md] $CID rep$REP already complete (prod.gro present) -- skipping"
    exit 0
fi
echo "[md] $(date) $CID rep$REP seed=$SEED"

if [[ "$USE_GPU" == "1" ]]; then
    nvidia-smi -L 2>&1 | head -4 || true
    # Full GPU-resident offload (single rank, 1 GPU/job): nonbonded + PME +
    # bonded + update/constraints all on the device, so the CPU only feeds it.
    # Forced `gpu` (not `auto`) on purpose -- if the build/card can't do a piece
    # we want a loud error, not a silent PME-on-CPU fallback (that silent
    # fallback is exactly what hid the SM-70 problem with dpflieger's binary).
    # PR + C-rescale pressure coupling and position restraints are all
    # supported with -update gpu in GROMACS 2024+.
    RUN_FLAGS="-nb gpu -pme gpu -bonded gpu -update gpu -ntmpi 1 -ntomp {cpus}"
    EM_FLAGS="-nb gpu -ntmpi 1 -ntomp {cpus}"
else
    RUN_FLAGS="-ntmpi 1 -ntomp {cpus}"
    EM_FLAGS="-ntmpi 1 -ntomp {cpus}"
fi

if [[ ! -f em.gro ]]; then
    "$GMX" grompp -f "$SYS/em.mdp" -c "$SYS/system.gro" -r "$SYS/system.gro" \\
        -p "$SYS/topol.top" -n "$SYS/index.ndx" -o em.tpr -maxwarn 2
    "$GMX" mdrun -deffnm em $EM_FLAGS
fi
if [[ ! -f nvt.gro ]]; then
    "$GMX" grompp -f "$SYS/nvt.mdp" -c em.gro -r em.gro -p "$SYS/topol.top" \\
        -n "$SYS/index.ndx" -o nvt.tpr -maxwarn 2
    "$GMX" mdrun -deffnm nvt $RUN_FLAGS
fi
if [[ ! -f npt.gro ]]; then
    "$GMX" grompp -f "$SYS/npt.mdp" -c nvt.gro -r nvt.gro -t nvt.cpt -p "$SYS/topol.top" \\
        -n "$SYS/index.ndx" -o npt.tpr -maxwarn 2
    "$GMX" mdrun -deffnm npt $RUN_FLAGS
fi
"$PY" "$SRC_DIR/write_prod_mdp.py" --seed "$SEED" --out prod.mdp ${{PROD_NSTEPS:+--nsteps $PROD_NSTEPS}}
if [[ ! -f prod.tpr ]]; then
    # -r: prod.mdp still has `define = -DPOSRES_CA` (weak CA tether, no membrane),
    # so grompp needs an explicit position-restraint reference since GROMACS 2018.
    "$GMX" grompp -f prod.mdp -c npt.gro -r npt.gro -t npt.cpt -p "$SYS/topol.top" \\
        -n "$SYS/index.ndx" -o prod.tpr -maxwarn 2
fi
if [[ -f prod.cpt ]]; then
    "$GMX" mdrun -deffnm prod -cpi prod.cpt $RUN_FLAGS
else
    "$GMX" mdrun -deffnm prod $RUN_FLAGS
fi
echo "[md] $(date) done $CID rep$REP"
echo "---- prod.log GPU/perf summary ----"
grep -aE "compatible GPUs|GPU will be used|will be executed on the GPU|on the GPU|PP task|PME task|Update task|Bonded interactions|Performance:" prod.log | sed 's/^/[md] /' || true
"""


def build_manifest(rows: list[dict], path: Path, replicas: int = config.N_REPLICAS) -> int:
    lines = []
    for r in rows:
        for rep in range(replicas):
            seed = 20260827 + rep * 101 + zlib.crc32(r["complex_id"].encode()) % 9973
            lines.append(f"{r['complex_id']}\t{rep}\t{seed}")
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="100 ps production + only the smoke complexes")
    ap.add_argument("--cpu", action="store_true", help="CPU-only: partition 'fast', no GRES, no GPU mdrun flags")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--gres", default=None)
    ap.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    ap.add_argument("--time", default=DEFAULT_TIME)
    ap.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    ap.add_argument("--conda-root", default=DEFAULT_CONDA_ROOT)
    ap.add_argument("--env", default=DEFAULT_ENV)
    ap.add_argument("--gromacs-module", default=DEFAULT_GROMACS_MODULE,
                    help="site module to `module load` for gmx; '' or 'none' to skip and use the conda gmx")
    ap.add_argument("--gmx-bin", default=DEFAULT_GMX_BIN,
                    help="explicit gmx binary path (overrides module/conda); '' to disable and use the module")
    ap.add_argument("--timing", action="store_true",
                    help="one-datapoint timing run: only the first smoke complex, replica 0, FULL 5 ns "
                         "production, %%1 -- to read ns/day off prod.log before committing the full array")
    args = ap.parse_args()

    partition = args.partition or (CPU_PARTITION if args.cpu else DEFAULT_PARTITION)
    gres = "" if args.cpu else (args.gres or DEFAULT_GRES)
    use_gpu = "0" if args.cpu else "1"
    gmod = "true" if args.gromacs_module.lower() in ("", "none") else args.gromacs_module
    gmx_bin = "" if args.gmx_bin.lower() in ("", "none") else args.gmx_bin

    rows = config.read_csv_rows(config.MANIFEST_CSV)
    if args.smoke or args.timing:
        rows = config.smoke_rows(rows)
    if not rows:
        print("[stage3] no manifest rows"); return

    ready = [r for r in rows if (config.SYSTEMS_DIR / r["complex_id"] / "prep.done").exists()]
    missing = [r["complex_id"] for r in rows if r not in ready]
    if missing:
        print(f"[stage3] {len(missing)} complex(es) not prepped (run prep_systems.py) -- excluded: "
              + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if not ready:
        print("[stage3] nothing ready to submit"); return

    replicas = config.N_REPLICAS
    prod_nsteps = ""
    maxc = args.max_concurrent
    tag = ""
    if args.timing:
        ready, replicas, maxc, tag = ready[:1], 1, 1, ".timing"   # 1 complex, rep0, full 5 ns
    elif args.smoke:
        prod_nsteps, tag = str(SMOKE_PROD_NSTEPS), ".smoke"

    manifest = config.SLURM_CFG_DIR / f"md_array_manifest{tag}.txt"
    n_tasks = build_manifest(ready, manifest, replicas)

    script = JOB_SCRIPT.format(
        partition=partition, cpus=args.cpus, mem=DEFAULT_MEM,
        gres_line=(f"#SBATCH --gres={gres}" if gres else ""),
        time=args.time, last=n_tasks - 1, maxc=maxc,
        logdir=config.SLURM_LOG_DIR, manifest=manifest, src_dir=Path(__file__).resolve().parent,
        systems_dir=config.SYSTEMS_DIR, md_dir=config.MD_DIR, conda_root=args.conda_root, env=args.env,
        prod_nsteps=prod_nsteps, use_gpu=use_gpu, gromacs_module=gmod, gmx_bin=gmx_bin,
    )
    script_path = config.SLURM_CFG_DIR / f"submit_md{tag}.sh"
    script_path.write_text(script)
    print(f"[stage3] {len(ready)} complex(es) x {replicas} replica(s) = {n_tasks} task(s) -> {manifest}")
    print(f"[stage3] partition={partition} gres={gres or '(none)'} gpu={use_gpu} "
          f"gmx-bin={gmx_bin or '(module/conda)'} conda-root={args.conda_root}")
    print(f"[stage3] array script: {script_path}  (--array=0-{n_tasks - 1}%{maxc})")

    if args.dry_run:
        print(f"[stage3] --dry-run: would `sbatch {script_path}`")
        return
    import subprocess
    subprocess.run(["sbatch", str(script_path)], check=True)
    print("[stage3] submitted")


if __name__ == "__main__":
    main()
