# HADDOCK3 redocking (GA1)

Physics-based cross-check on ABCfold's ab initio cofolding: does HADDOCK3 --
a restraint-driven docking engine scored by real physics (van der Waals /
electrostatics / restraint energy, not a learned potential) -- converge on
a similar Gibberellin A1 (GA1) pose in a confirmed importer, and fail to
find a stable pocket-bound pose in a confirmed non-importer? Agreement or
disagreement on both fronts is the actual signal about whether the ab
initio backends in this pipeline are learning real binding physics or
memorizing training-data poses.

Generalized in spirit from `rescoring/` (PyRosetta energetics) -- same
"new physics-based cross-check subrepo" shape and `data/`/`src/`/`results/`
layout, reusing this pipeline's own outputs and the sibling
`NPF_pocket_pipeline`'s CDD pocket annotations rather than recomputing any
of it.

## Scope

One receptor per **(protein, macro-conformation `ca_cluster`)** for every
NPF_LDA_kernel importer/non-importer that has both CDD pocket residues and
usable clustering output -- not one pose per protein. Every protein's
`pca_k3` macro-state clustering (`worflows/postprocessing`'s own TM-
alignment + PCA-k3 stage, pools apo and holo frames on one shared
coordinate frame) currently yields exactly 3 clusters, so:

- **5 of 12** NPF_LDA_kernel `hc_importers` are usable (GA1-holoform pose
  + CDD residues both present): NPF3.1, NPF2.12, NPF2.13, NPF2.10, NPF2.5.
  The other 7 were either co-folded with a *different* ligand in this
  pipeline (NPF2.7/NPF2.3/NPF2.4/NPF1.1/NPF1.2 -> nitrate, NPF4.2 -> ABA
  -- see `scripts/ligand_assignment.py`'s `ligand_for()`) or have a GA1
  pose but a genuinely empty CDD/InterProScan result (NPF4.1).
- **19 of 21** `hc_non_importers` are usable (apoform structure + CDD
  residues); NPF8.5/NPF5.9 have the same genuine-empty-CDD-result issue.
- **5 x 3 + 19 x 3 = 72 total complexes** (15 importer, 57 non-importer).

Receptor source is `results/tm_reannotated/<protein>/pca_k3/` (a curated,
materialized subset of frames per cluster), **not** raw
`results/abcfold/` or `rescoring/data/manifest.csv` -- confirmed by hand
that a meaningful fraction of raw per-frame CIFs get deleted by this
pipeline's own storage-compression step once it's actually run on the
cluster, while `tm_reannotated`'s `symlinked == True` frames survive that.

## Setup

```bash
conda env create -f ../envs/redocking.yaml
conda activate redocking
```

`haddock3` bundles a precompiled CNS binary automatically on `pip
install` -- confirmed working with no recompile needed on this cluster.
No GPU needed -- CNS is CPU-only.

## Pipeline

```bash
cd src
python build_ga1_from_ga3.py        # Stage 0: GA1 3D structure, templated on GA3's real CCD geometry
python prep_ligand_topology.py      # Stage 1: GA1 CNS topology/params (BioExcel AcpypeParamsCNS)
python make_manifest.py             # Stage 2: 72-complex manifest, auto-filtered by data availability
python extract_receptor_pdb.py      # Stage 3: protein-only receptor PDB per manifest row
python define_active_passive.py     # Stage 4: CDD active residues -> haddock3-restraints -> ambig.tbl
python make_haddock_cfg.py --ncores 32 --tolerance 50   # Stage 5: per-complex run.cfg
python run_haddock_batch.py --cpus 32 --mem 128G        # Stage 6: ONE SLURM job array (IFB)
python compare_to_abcfold.py        # Stage 7: RMSD vs. ABCfold pose (importer) / pocket engagement (non-importers)
```

### GA1 ligand structure (Stage 0)

GA1 and GA3 (Gibberellin A3) differ only in ring-A saturation, so most of
the molecule is structurally identical. `build_ga1_from_ga3.py` templates
GA1's geometry on GA3's own **real** crystal structure: downloads GA3's
RCSB Chemical Component Dictionary entry (correct bond table + real
"model" coordinates, not the idealized Corina conformer), finds the
maximum common substructure with GA1's SMILES (~84% of heavy atoms), then
`AllChem.ConstrainedEmbed`s GA1 tethered to GA3's real coordinates.
Confirmed on a real run: 21/25 heavy atoms matched, tether RMSD 0.047 A,
geometry sanity check clean.

### Ligand topology (Stage 1)

`prep_ligand_topology.py` uses the actual BioExcel Building Block for
this -- `biobb_chemistry.acpype.acpype_params_cns.AcpypeParamsCNS`.
**Needs `acpype` (PyPI) and AmberTools (conda-forge `ambertools`,
antechamber/sqm) installed separately** -- `biobb_chemistry` only wraps
the `acpype` CLI, it does not vendor either. Also stamps a chain ID onto
OpenBabel's PDB output (blank by default; HADDOCK3 requires one).
Confirmed working end-to-end, including a standalone `[topoaa]`
validation pass before the topology is trusted in a real docking config.

### Manifest (Stage 2)

`make_manifest.py` builds the full candidate list from
`NPF_LDA_kernel`'s `hc_importers`/`hc_non_importers`, filters by
`ligand_for()` (importers only -- must actually have been co-folded with
GA1 in this pipeline), CDD residue availability, and usable
`tm_reannotated` clusters, then picks one representative frame per
`(protein, ca_cluster)` (highest `ptm`, deterministic tiebreak). Prints a
skip report explaining every excluded candidate.

### Active/passive residues (Stage 4)

"Active" = CDD/InterPro putative pocket residues `NPF_pocket_pipeline`
already computed (read directly, never recomputed here). "Passive" is
derived with HADDOCK3's own `haddock3-restraints passive_from_active` CLI,
then `active_passive_to_ambig` turns both into the `.tbl` AIR file.
**CLI contract confirmed by hand, not guessable from `--help`**:
`passive_from_active`'s active-residue list is a literal comma-separated
CLI argument, not a file; `active_passive_to_ambig` needs one combined
file per molecule with exactly 2 lines (active, then passive).

### Docking protocol (Stage 5)

`topoaa -> rigidbody -> caprieval -> seletop -> flexref -> caprieval`.
**Deliberately missing the reference example's `ilrmsdmatrix ->
clustrmsd -> seletopclusts` clustering tail** -- `fast-rmsdmatrix`
(bundled with this pip-installed HADDOCK3) needs glibc 2.38, but every
compute node on this cluster runs Ubuntu 20.04 (glibc 2.31); only the
login node has a new enough glibc, which is why running it there directly
gave a false impression of working. Per the user: RMSD/clustering isn't
needed from HADDOCK3 itself and can be computed post-hoc from the kept
model PDBs, so the whole clustering tail was dropped rather than chasing
a binary-compatibility fix. The final `[caprieval]` ranks every kept
model by HADDOCK score directly -- no `cluster_id`/`cluster_ranking`
columns in `capri_ss.tsv`.

Two more per-cfg fixes, both confirmed necessary the hard way:
- `ncores` (HADDOCK3's own CNS-job parallelism) defaults to 4 and does
  **not** read the SLURM `--cpus-per-task` allocation -- set explicitly
  at the top level of the cfg, kept in sync with `run_haddock_batch.py`'s
  `--cpus`.
- `tolerance` (per-module failure-tolerance %, default 5) needs raising
  (`tolerance = 50`) at production job counts -- a real run reported 30%
  of `[flexref]` output "not generated" despite every file actually being
  present and valid (a filesystem-metadata-visibility lag on this
  cluster's shared storage, not a real per-job failure).

`ligand_top_fname`/`ligand_param_fname` must be restated in **every**
CNS-running module's own section (`[rigidbody]`, `[flexref]`), not just
`[topoaa]` -- HADDOCK3 only auto-propagates them from `[topoaa]` in the
`autotoppar` branch.

### Submission (Stage 6)

One SLURM **job array** (`--array=0-71%8`, throttled to 8 concurrent
tasks), not 72 individual `sbatch` calls -- cleaner to monitor/cancel as
one unit at this scale. Each array task reads its own line from
`results/haddock_runs/_cfgs/array_manifest.txt` via
`$SLURM_ARRAY_TASK_ID`. `--dry-run` shows what would be submitted without
touching SLURM.

### Comparison (Stage 7)

- **Importer**: ligand heavy-atom RMSD between HADDOCK3's top-ranked
  model and ABCfold's own predicted GA1 pose, after Kabsch-superposing
  the two receptors' C-alpha atoms.
- **Non-importers**: no ABCfold ligand pose exists to compare against --
  instead, for each of the top-4 models by HADDOCK score (no clustering,
  so these can include near-identical poses), how many CDD active-pocket
  residues does the docked GA1 actually contact (<= 4.5 A)? Low/no
  engagement is *consistent with* (not proof of) the non-importer
  classification.

`LIGAND_CHAIN_HADDOCK = "B"` (molecule 2 in `molecules=[receptor,
ligand]`) is confirmed correct against a real run's own log.

## What's verified

Every stage has been run for real against this pipeline's actual data,
not just syntax-checked, including a full 72-complex production array on
the cluster with the current (no-clustering) protocol -- first 8-task
wave confirmed COMPLETED with genuine, varying HADDOCK scores and DockQ
values. `compare_to_abcfold.py`'s logic has been updated for the current
protocol but not yet run against the full array's completed output.

Full blow-by-blow of every bug found and fixed (HADDOCK3 config gotchas,
the glibc incompatibility investigation, the login-node-vs-compute-node
false-positive lesson) is in project memory --
`reference_haddock3_cns_ligand_param_propagation` and
`project_redocking_pipeline_plan` -- read those before touching this
pipeline again rather than re-deriving any of it from scratch.

## Repo layout

```text
redocking/
  data/
    ga1_from_ga3.sdf / .log       Stage 0 output
    manifest.csv                  Stage 2 output (72 complexes)
    receptors/                    Stage 3 output (protein-only PDB per complex)
    restraints/                   Stage 4 output (active/passive lists + ambig.tbl per complex)
    _cache/                       downloaded CCD files, standardized ligand PDB, acpype work dir
  ligand_topology/
    GA1_cns.top / GA1_cns.param   Stage 1 output
  src/
    config.py                     shared paths/constants
    build_ga1_from_ga3.py         Stage 0
    prep_ligand_topology.py       Stage 1
    make_manifest.py              Stage 2
    extract_receptor_pdb.py       Stage 3
    define_active_passive.py      Stage 4
    make_haddock_cfg.py           Stage 5
    run_haddock_batch.py          Stage 6 (SLURM job array, IFB)
    compare_to_abcfold.py         Stage 7
  results/
    haddock_runs/
      _cfgs/                       generated .cfg + array_manifest.txt + submit_array.sh
      _slurm_logs/                 per-task SLURM stdout/stderr (task_<N>.log/.err)
      <complex_id>/                haddock3's own native run-directory output, one per complex
    comparison/                   Stage 7 output (per-complex JSON + summary.csv)
```
