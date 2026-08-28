# GROMACS on pangloss — resolution + diagnosis

**Status (2026-08-28): resolved, no admin action needed.** GROMACS comes from
the user Conda env; see "Resolution" below. This file keeps the record of the
GPU-detection problem that got us here, in case anyone revisits it or wants to
provide a site module later.

---

## Resolution

The `mmgbsa` Conda env carries its own CUDA GROMACS:

```bash
CONDA_OVERRIDE_CUDA=12.4 mamba install -n mmgbsa -c conda-forge "gromacs=*=nompi_cuda_*"
```

`CONDA_OVERRIDE_CUDA` is required because the install runs on a login node with
no GPU, so the solver would otherwise pick the CPU build. The pipeline then
uses `~/miniforge3/envs/mmgbsa/bin/gmx` for every stage (grompp, mdrun,
`gmx_MMPBSA`) — one GROMACS version end to end, self-contained (bundles its own
CUDA runtime + `share/gromacs/top`).

```
GROMACS version:  2026.3-conda_forge
GPU support:      CUDA
CUDA targets:     50;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;120
```

`70-real` covers niobe's V100S; the bare `50` and `120` add PTX so any other
card also works.

---

## The problem this replaced

IT (dpflieger) kindly hand-built a CUDA GROMACS at
`/shared/home/dpflieger/Shared/gmx` (GROMACS 2026.3, source tree
`/shared/home/dpflieger/Softs/gromacs-2026.3/build`). It runs, and
`gmx --version` reports `GPU support: CUDA` — but on niobe every `mdrun`
**silently fell back to the CPU**.

### niobe's GPUs

```
$ nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
name, compute_cap, driver_version
Tesla V100S-PCIE-32GB, 7.0, 580.126.20
Tesla V100S-PCIE-32GB, 7.0, 580.126.20
```

Compute capability **7.0** (SM 70).

### The site binary's GPU code

```
$ /shared/home/dpflieger/Shared/gmx --version | grep -i cuda
GPU support:   CUDA
CUDA targets:  75;80-real;86-real;89-real;90-real;100-real;120
```

**No 7.0**, and every entry except `120` carries `-real` (SASS only, no PTX
fallback). So the binary has no GPU code the V100S can run.

### The silent symptom

In the `mdrun` log (`nvt.log` / `md.log`) — no error, one quiet line:

```
Running on 1 node with total 48 cores, 96 processing units, 0 compatible GPUs
  GPU info:
    Number of GPUs detected: 1
Using 16 OpenMP threads
```

`0 compatible GPUs` → the run proceeds entirely on 16 CPU threads. From the
outside the job looks normal, it is just ~8× slower. Measured on this fallback:
**~25 ns/day** for the ~106k-atom NPF3.1 + GA1 system (vs. the GPU number in
`RESULTS.md`).

---

## If a site module is ever wanted

Not necessary, but if IT wants to provide one so other users benefit: rebuild
with **`70`** in the target list and keep the PTX layer —

```
cd /shared/home/dpflieger/Softs/gromacs-2026.3/build
cmake -DGMX_CUDA_TARGET_SM="70;75;80;86;89;90" ..   # no "-real" -> PTX kept
make -j gmx
```

~15–40 min (only CUDA kernels recompile). Verify:
`gmx --version` shows `70` in `CUDA targets`, and an `mdrun` on niobe logs
`1 GPU selected` / tasks `on the GPU`, not `0 compatible GPUs`. Then point the
pipeline at it with `run_md_batch.py --gmx-bin <path>` (or expose it as an Lmod
module named `gromacs` and use `--gmx-bin '' --gromacs-module gromacs`).
