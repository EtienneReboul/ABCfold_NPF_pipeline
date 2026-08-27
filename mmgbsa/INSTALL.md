# Software request for `mmgbsa/` (hand this to IT)

The `mmgbsa/` pipeline needs **one** piece of software installed as a module on
**pangloss** (`pangloss.ibmp.unistra.fr`). Everything else comes from a user
Conda environment (`~/miniforge3`) and needs no admin action.

## Requested: a GPU-enabled GROMACS module

- **Package**: GROMACS, version **2024.x or 2025.x** (any recent stable).
- **Build**: single precision, **CUDA-enabled**, thread-MPI (regular MPI not
  required — every run is single-node). `-DGMX_BUILD_OWN_FFTW=ON` is fine if
  FFTW isn't already available.
- **GPU architectures**: the jobs run on the `cryoem` partition (node
  `oss117`, 2× NVIDIA **L4**, compute capability **8.9**) and possibly the
  `gpu` partition (node `niobe`, 2× Tesla). A default recent-CUDA build
  (CUDA ≥ 12.0) that targets SM 7.0 through 8.9 covers both.
- **Expose as an Lmod module** named `gromacs` (or `gromacs/<version>`),
  in the same modulefile tree as the existing bio tools
  (`/shared/biotools/modules/...`), so that on a login or compute node:

  ```bash
  module load gromacs
  gmx --version        # must report:  GPU support:  CUDA
  ```

That's the whole request. If it's easier to also provide an `AmberTools`
module (≥ 22), that's welcome but **not needed** — the pipeline installs
AmberTools into its Conda env.

## Why (context, not required reading)

Short classical-MD runs (a few ns each, ~50–70k-atom protein–ligand–water
systems) followed by MM-GBSA end-state free-energy analysis. GPU offload
(`mdrun -nb gpu`) is ~5–10× faster than CPU for this system size; the L4 node
is the target. Without a module the pipeline falls back to a CPU-only
Conda GROMACS, which works but is much slower for the full ~140-complex set.

## After it's installed

Tell the user (`ereboul`) the exact module name. Nothing else changes on the
admin side — the user creates the Conda env and submits the SLURM job arrays.
