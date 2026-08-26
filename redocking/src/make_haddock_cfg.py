"""
redocking/src/make_haddock_cfg.py
====================================
Stage 5: per-complex HADDOCK3 .cfg, adapted from haddock3's own
`examples/docking-protein-ligand/docking-protein-ligand-full.cfg` template.
Module order: topoaa -> rigidbody -> caprieval -> seletop -> flexref ->
caprieval -- **deliberately missing the reference example's ilrmsdmatrix
-> clustrmsd -> seletopclusts clustering tail**, see below.

Two ligand-topology modes, selected with --autotoppar:
  - default: ligand_top_fname/ligand_param_fname point at
    ligand_topology/GA1_cns.top / GA1_cns.param (prep_ligand_topology.py's
    BioExcel/acpype output).
  - --autotoppar: HADDOCK3's [topoaa] autotoppar=true instead -- generates
    ligand topology directly from data/ga1_from_ga3.sdf-derived PDB, no
    acpype dependency. Fast cross-check / fallback if the acpype route
    isn't validated yet (see prep_ligand_topology.py's module docstring).

**No clustering tail -- root cause found, not worth working around
(2026-08-25).** `[ilrmsdmatrix]` crashed deterministically at any ncores
setting (`FileNotFoundError: ilrmsd_0.matrix`, no tolerance mechanism
unlike the CNS modules) -- an `ncores = 1` override initially looked like
a fix (via `haddock3 --restart <step>` on a failed run) but that "success"
turned out to be an artifact of running the restart by hand on the LOGIN
node, not a real fix: a fresh run through the SLURM array failed
identically even with `ncores = 1`. The actual cause, confirmed via the
SLURM task's own `.err` file (HADDOCK3's per-job `.out` capture doesn't
redirect the underlying binary's stderr, so it never appears there):
`fast-rmsdmatrix: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.38'
not found`. This cluster's login node runs Ubuntu 24 (glibc 2.39) but
every compute node runs Ubuntu 20.04 (glibc 2.31) -- confirmed via a
throwaway `sbatch` job checking `ldd --version` / `/etc/os-release` on an
actual compute node. `fast-rmsdmatrix` (bundled with this pip-installed
HADDOCK3) simply cannot run anywhere real jobs execute on this cluster.
Per the user (2026-08-25): RMSD/clustering isn't needed from HADDOCK3
itself -- it can be computed post-hoc from the kept model PDBs with any
tool, so the pragmatic fix is dropping the whole clustering tail rather
than chasing a binary-compatibility fix (recompiling from source, or
escalating to IFB about the cluster's login/compute glibc mismatch,
would both be far more effort for a step that was only ever doing
presentation-layer deduplication, not part of the actual physics).
The final `[caprieval]` right after `[flexref]` already ranks every kept
model by HADDOCK score directly -- `compare_to_abcfold.py` selects
top-N models from THAT ranking now, not from cluster representatives.

**Confirmed by hand at production scale (2026-08-25)**: `[rigidbody]`/
`[flexref]`'s per-module `tolerance` parameter (default 5%, max 99) --
the max percentage of a step's CNS jobs allowed to produce no output
before HADDOCK3 aborts the whole run -- needs raising well above its
default on this cluster's shared filesystem. Hit a real run where
`[flexref]` (40 jobs) reported "30.00% of output was not generated",
but every one of the 40 expected `flexref_N.pdb` files was actually
present, correctly sized, with a real HADDOCK score, and `io.json`
itself recorded 40/40 -- `haddock.libs.libontology.ModuleIO.check_faulty()`
checks `Path.exists()` immediately after the parallel engine reports all
jobs finished, before the shared-storage filesystem had made every file's
metadata visible yet (a stat-cache/NFS-propagation lag, not a real
per-job failure) -- confirmed by re-inspecting the same files afterward
and finding them all intact. Also note `check_faulty()` calls
`remove_missing()` regardless of whether the tolerance check itself
passes, so a raised tolerance lets the run continue with (rarely) a
handful of models genuinely dropped, not silently duplicated or
corrupted.

**Confirmed by hand against a real run (2026-08-25) -- NOT guessable from
the fetched upstream example alone**: `ligand_top_fname`/`ligand_param_fname`
set only under `[topoaa]` are NOT automatically carried through to later
CNS-running modules ([rigidbody], [flexref], ...) for the manual/acpype
route. HADDOCK3's own topoaa source
(`haddock/modules/topology/topoaa/__init__.py`) only writes these into
`self._output_params` (the mechanism that auto-propagates them onto every
downstream module) inside the `autotoppar` branch, for single-model
inputs -- the plain `ligand_top_fname` branch does not. Concretely: a
`rigidbody` run with only `[topoaa]`'s ligand params set gets an EMPTY
`$ligand_param_fname` in its own generated .inp (confirmed by reading the
decompressed `*.inp`/`*.out.gz` from a `debug = true` run), so CNS never
loads the ligand's GAFF nonbonded (Lennard-Jones) parameters and aborts
with `%NBUPDA-ERR: missing nonbonded Lennard-Jones parameters` the moment
the energy term needs them. **Every module in `CNS_LIGAND_MODULES` below
needs its own `ligand_top_fname`/`ligand_param_fname` lines**, not just
`[topoaa]` -- confirmed this is genuinely module-scoped (each of
rigidbody/flexref/emref/mdref/etc. has its own `ligand_param_fname` entry
in its own `defaults.yaml`, defaulting to empty string). Not needed for
`--autotoppar`, since that branch's own propagation already covers it for
this pilot's single-model (non-ensemble) case.

**Confirmed by hand (2026-08-25)**: HADDOCK3's `ncores` (how many CNS jobs
run in parallel within a module) is a GLOBAL cfg parameter
(`haddock/modules/defaults.yaml`, not any one module's own defaults.yaml)
that **defaults to 4** and is completely independent of the SLURM
`--cpus-per-task` a job was submitted with -- HADDOCK3 does not read the
SLURM allocation. Leaving it unset meant the pilot's first successful run
processed rigidbody's 200 CNS jobs 4-at-a-time despite a 32-core SLURM
request (`Selected 4 cores to process 200 jobs, with 64 maximum available
cores` in its own log) -- confirmed by hand, not a guess. Set explicitly
at the top level of the cfg (outside any `[module]` section, like
`run_dir`/`molecules`) so it applies to every module.

**Confirmed by hand (2026-08-25)**: the generated `.cfg` must NOT live
inside `run_dir` itself. HADDOCK3 refuses to write into a `run_dir` that
already exists and is non-empty (`--restart` required otherwise) -- a run
whose only pre-existing content was its own `run.cfg` still failed with
that same "exists and is not empty" error, in ~90 seconds, on all 3 pilot
SLURM jobs the first time this was tried. Cfg files live in a sibling
`_cfgs/` directory instead; `run_dir` itself must stay untouched/absent
until HADDOCK3 creates it.

Output: results/haddock_runs/_cfgs/<complex_id>.cfg (the `run_dir =`
line inside it points at results/haddock_runs/<complex_id>/, which
HADDOCK3 creates fresh on execution -- never pre-create or write into it).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import config
from extract_receptor_pdb import RECEPTOR_DIR
from define_active_passive import RESTRAINTS_DIR

# Every module downstream of [topoaa] that runs its own CNS energy
# evaluation on the complex needs the ligand's custom parameters restated
# -- see module docstring. caprieval/seletop don't run CNS minimization
# and don't take this parameter.
CFG_TEMPLATE = """\
# Generated by redocking/src/make_haddock_cfg.py -- adapted from haddock3's
# examples/docking-protein-ligand/docking-protein-ligand-full.cfg
run_dir = "{run_dir}"
molecules = ["{receptor_pdb}", "{ligand_pdb}"]
ncores = {ncores}

[topoaa]
{topoaa_ligand_block}

[rigidbody]
{ligand_block}
ambig_fname = "{ambig_tbl}"
sampling = 200
tolerance = {tolerance}

[caprieval]
reference_fname = "{receptor_pdb}"

[seletop]
select = 40

[flexref]
{ligand_block}
ambig_fname = "{ambig_tbl}"
tolerance = {tolerance}

[caprieval]
"""

LIGAND_PARAM_BLOCK = """\
ligand_top_fname = "{top}"
ligand_param_fname = "{param}"
"""

TOPOAA_AUTOTOPPAR_BLOCK = """\
autotoppar = true
"""


DEFAULT_NCORES = 32  # keep in sync with run_haddock_batch.py's DEFAULT_CPUS -- ncores has no way
                      # to auto-detect the SLURM --cpus-per-task a job runs under (see module docstring)
DEFAULT_TOLERANCE = 50  # generous headroom over the observed ~30% transient filesystem-visibility
                         # lag at production scale (see module docstring) -- still catches a truly
                         # catastrophic (>50%) per-job failure rate, doesn't mask real breakage


def make_cfg(complex_id: str, receptor_pdb: Path, ligand_pdb: Path, ambig_tbl: Path,
             run_dir: Path, out_cfg: Path, autotoppar: bool, ncores: int = DEFAULT_NCORES,
             tolerance: int = DEFAULT_TOLERANCE) -> Path:
    if autotoppar:
        topoaa_block = TOPOAA_AUTOTOPPAR_BLOCK
        ligand_block = ""  # autotoppar's own single-model propagation covers rigidbody/flexref already
    else:
        if not (config.GA1_CNS_TOP.exists() and config.GA1_CNS_PARAM.exists()):
            raise FileNotFoundError(
                f"{config.GA1_CNS_TOP} / {config.GA1_CNS_PARAM} not found -- run "
                f"prep_ligand_topology.py first, or pass --autotoppar to use HADDOCK3's "
                f"built-in ligand-topology generation instead."
            )
        ligand_block = LIGAND_PARAM_BLOCK.format(top=config.GA1_CNS_TOP, param=config.GA1_CNS_PARAM)
        topoaa_block = ligand_block

    cfg_text = CFG_TEMPLATE.format(
        run_dir=run_dir,
        receptor_pdb=receptor_pdb,
        ligand_pdb=ligand_pdb,
        ambig_tbl=ambig_tbl,
        topoaa_ligand_block=topoaa_block,
        ligand_block=ligand_block,
        ncores=ncores,
        tolerance=tolerance,
    )
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(cfg_text)
    return out_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autotoppar", action="store_true",
                         help="Use HADDOCK3's built-in autotoppar ligand-topology generation "
                              "instead of prep_ligand_topology.py's acpype/BioExcel output.")
    parser.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                         help="HADDOCK3's own parallel-CNS-jobs count -- match whatever "
                              "--cpus you'll pass run_haddock_batch.py, they don't sync automatically.")
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE,
                         help="Per-module failure tolerance percentage -- raised above HADDOCK3's "
                              "5%% default to absorb this cluster's shared-filesystem visibility lag "
                              "at production job counts (see module docstring).")
    args = parser.parse_args()

    if not config.GA1_FROM_GA3_SDF.exists():
        raise FileNotFoundError(f"{config.GA1_FROM_GA3_SDF} not found -- run build_ga1_from_ga3.py first.")
    # HADDOCK3's molecules= list takes PDB, not SDF -- reuse the same
    # OpenBabel standardization prep_ligand_topology.py already does for
    # acpype's input, so both topology modes consume the exact same ligand
    # geometry/atom set.
    from prep_ligand_topology import standardize_with_openbabel
    ligand_pdb = config.DATA_DIR / "_cache" / "GA1_standardized.pdb"
    ligand_pdb.parent.mkdir(parents=True, exist_ok=True)
    standardize_with_openbabel(config.GA1_FROM_GA3_SDF, ligand_pdb)

    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))

    cfgs_dir = config.HADDOCK_RUNS_DIR / "_cfgs"
    for row in rows:
        complex_id = row["complex_id"]
        receptor_pdb = RECEPTOR_DIR / f"{complex_id}_receptor.pdb"
        ambig_tbl = RESTRAINTS_DIR / f"{complex_id}_ambig.tbl"
        run_dir = config.HADDOCK_RUNS_DIR / complex_id
        out_cfg = cfgs_dir / f"{complex_id}.cfg"
        make_cfg(complex_id, receptor_pdb, ligand_pdb, ambig_tbl, run_dir, out_cfg,
                 args.autotoppar, args.ncores, args.tolerance)
        print(f"{complex_id}: wrote {out_cfg}")


if __name__ == "__main__":
    main()
