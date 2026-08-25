# HADDOCK3 redocking (GA1 pilot)

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
layout, reusing this pipeline's own outputs (`results/abcfold/`,
`rescoring/data/manifest.csv`) and the sibling `NPF_pocket_pipeline`'s CDD
pocket annotations rather than recomputing any of it.

**Pilot scope** (see plan discussion): 1 confirmed HC importer
(`NPF2.10_Q944G5`, cluster-representative GA1-holoform pose) + 2 confirmed
non-importers (`NPF6.1_Q9LYR6`, `NPF8.1_Q9M390`, apoform ABCfold
structures -- all 21 `NPF_LDA_kernel` non-importers already have apoform
output in this pipeline's `results/abcfold/`, since this pipeline's
UniProt query pulls the whole Arabidopsis NRT1/PTR family and runs
anything not ligand-assigned as apoform).

## Setup

```bash
conda env create -f ../envs/redocking.yaml
conda activate redocking
```

`haddock3` bundles a precompiled CNS binary automatically on `pip
install`. If it errors on this cluster's architecture, see
`docs/CNS.md` in the haddock3 repo for the manual recompile
(gcc/gfortran/csh -- already in `envs/redocking.yaml`) + binary-copy steps.
No GPU needed -- CNS is CPU-only.

## Pipeline

```bash
cd src
python build_ga1_from_ga3.py        # Stage 0: GA1 3D structure, templated on GA3's real CCD geometry
python make_manifest.py             # Stage 2: pilot manifest (1 importer + 2 non-importers)
python extract_receptor_pdb.py      # Stage 3: protein-only receptor PDB per manifest row
python define_active_passive.py     # Stage 4: CDD active residues -> haddock3-restraints -> ambig.tbl
python prep_ligand_topology.py      # Stage 1: GA1 CNS topology/params (BioExcel AcpypeParamsCNS)
python make_haddock_cfg.py          # Stage 5: per-complex run.cfg (add --autotoppar to skip Stage 1's output)
python run_haddock_batch.py --dry-run   # Stage 6: SLURM submission (IFB) -- drop --dry-run once partition/account confirmed
python compare_to_abcfold.py        # Stage 7: RMSD vs. ABCfold pose (importer) / pocket engagement (non-importers)
```

Stage numbering intentionally doesn't run 0,1,2,3... in file order: Stage 1
(ligand topology) is independent of Stages 2-4 (receptor/restraint prep)
and only needed before Stage 5, so it's listed after them above to
group "things you can run immediately" first.

### GA1 ligand structure (Stage 0)

GA1 and GA3 (Gibberellin A3) differ only in ring-A saturation -- GA3 has a
conjugated ene-lactone ring A, GA1's is saturated with a 3-beta-OH -- so
most of the molecule (bicyclic lactone bridge, exocyclic-methylene D-ring,
carboxylic acid) is structurally identical. Rather than a bare RDKit
ETKDG embed from GA1's SMILES alone (config.yaml's `ligands.GA1.smiles`),
`build_ga1_from_ga3.py` templates GA1's geometry on GA3's own **real**
crystal structure:

1. Download GA3's RCSB Chemical Component Dictionary entry
   (`https://files.rcsb.org/ligands/download/GA3.cif`) -- this carries
   both an explicit, correct bond table (not distance-perceived) AND real
   "model" coordinates from an actual deposited structure the CCD entry
   cites (`pdbx_model_coordinates_db_code`, currently PDB `3ED1`) --
   **not** the CCD's separate "ideal" Corina-generated conformer, which
   has no real ring pucker.
2. Find the maximum common substructure (MCS) between GA3 and GA1
   (typically ~84% of GA1's heavy atoms, everything except ring A).
3. `AllChem.ConstrainedEmbed`: embed GA1 with the MCS-matched atoms
   tethered to GA3's real coordinates, MMFF-relaxing the rest freely.
4. Sanity-check (bond lengths vs. covalent-radii sums, no non-bonded heavy
   -atom clashes) before writing `data/ga1_from_ga3.sdf` +
   `data/ga1_from_ga3.log` (MCS coverage, tether RMSD, sanity-check
   result -- provenance record).

Confirmed on a real run: 21/25 GA1 heavy atoms matched (84%), tether RMSD
0.047 A, sanity check clean.

### Ligand topology (Stage 1)

`prep_ligand_topology.py` uses the actual BioExcel Building Block for
this -- `biobb_chemistry.acpype.acpype_params_cns.AcpypeParamsCNS` (a
Python-wrapped acpype run producing CNS/XPLOR-format `.top`/`.par`
directly), not the raw acpype CLI. Input is `data/ga1_from_ga3.sdf`
standardized/protonated via OpenBabel first.

**Known risk, not yet resolved**: acpype's CNS output naming/format
conventions are not guaranteed to match what HADDOCK3's `[topoaa]` module
expects out of the box. `validate_against_topoaa` runs a standalone
`[topoaa]` on the ligand alone specifically to catch this cheaply, before
it's ever used in a real docking config -- treat a failure there as
expected-possible, not a bug in this script, and budget time to patch the
CNS output (same category of fix `rescoring/src/prep_ligand.py`'s
`ADD_RING` renumbering was for PyRosetta params). If this stalls,
`make_haddock_cfg.py --autotoppar` bypasses it entirely via HADDOCK3's own
built-in ligand-topology generation (`[topoaa] autotoppar = true`) as a
fast cross-check.

### Active/passive residues (Stage 4)

"Active" residues are the CDD/InterPro putative pocket residues
`NPF_pocket_pipeline` already computed
(`../../NPF_pocket_pipeline/data/interpro/cdd_summary.json` -- read
directly, never recomputed here; all 3 pilot proteins already have this).
"Passive" residues are derived with HADDOCK3's own
`haddock3-restraints passive_from_active` CLI (surface/neighbor-expands
the active list against the real receptor structure) rather than a
hand-rolled distance-cutoff script, then
`haddock3-restraints active_passive_to_ambig` turns both into the
`.tbl` AIR file `[rigidbody]`/`[flexref]` consume.

### Docking protocol (Stage 5)

Adapted from HADDOCK3's own
`examples/docking-protein-ligand/docking-protein-ligand-full.cfg` --
module order `topoaa -> rigidbody -> caprieval -> seletop -> flexref ->
ilrmsdmatrix -> clustrmsd -> seletopclusts -> caprieval`. **Diff
`make_haddock_cfg.py`'s `CFG_TEMPLATE` against the real upstream example
file the first time this env is set up** -- it was adapted from a fetched
summary of that file, not a byte-for-byte copy, so per-parameter defaults
(sampling counts, cluster count, etc.) are reasonable starting points, not
independently verified against the original.

### Comparison (Stage 7)

- **Importer**: ligand heavy-atom RMSD between HADDOCK3's top-ranked model
  and ABCfold's own predicted GA1 pose, after Kabsch-superposing the two
  receptors' C-alpha atoms (HADDOCK3's `flexref` step allows some backbone
  movement, so this can't assume the two structures are already aligned).
- **Non-importers**: no ABCfold ligand pose exists to compare against
  (apoform) -- instead, for each ranked cluster representative, how many
  CDD active-pocket residues does the docked GA1 actually contact
  (<= 4.5 A)? Low/no engagement is *consistent with* (not proof of) the
  non-importer classification.

**Not yet run against a real HADDOCK3 output directory** -- `compare_to_abcfold.py`'s
`LIGAND_CHAIN_HADDOCK = "B"` assumption (molecule 2 in
`molecules=[receptor, ligand]` gets chain "B" with no explicit segid
override) needs confirming against an actual run's output PDB before
trusting any number this script produces.

## What's verified vs. not

**Env built and the entire pipeline mechanically validated end-to-end on
the IFB cluster (2026-08-25)**, `envs/redocking.yaml` built there via
`mamba env create` (plus `acpype` + `ambertools` added afterward -- see
below), CNS binary confirmed executable with no recompile needed. A
4-sample smoke test (`sampling=4`, not a real pilot run) completed the
full `topoaa -> rigidbody -> caprieval -> seletop -> flexref -> caprieval
-> ilrmsdmatrix -> clustrmsd -> seletopclusts -> caprieval` protocol with
real, varying HADDOCK scores and cluster output. Stages 0/2/3/4 (GA1
build, manifest, receptor extraction, restraints) and Stage 1 (BioExcel
`AcpypeParamsCNS` ligand topology, validated against a standalone
`[topoaa]` run) all ran for real against this pipeline's actual data, not
just syntax-checked.

**Two real bugs were found and fixed by actually running this** (not
guessable from HADDOCK3's docs alone -- see
`reference_haddock3_cns_ligand_param_propagation` in project memory for
full detail if picking this up again):
1. `haddock3-restraints passive_from_active`'s `active_list` argument is a
   literal comma-separated string, not a file path; `active_passive_to_ambig`
   needs one combined "actpass" file per molecule (active on line 1,
   passive on line 2, exactly 2 lines) -- `define_active_passive.py` fixed.
2. `ligand_top_fname`/`ligand_param_fname` set only under `[topoaa]` do
   NOT propagate to `[rigidbody]`/`[flexref]` for the manual (non-
   autotoppar) route -- each module needs its own copy of both lines, or
   CNS aborts deep in energy minimization with no clear top-level error
   (`%NBUPDA-ERR: missing nonbonded Lennard-Jones parameters`, only
   visible by setting that module's own `debug = true` and reading the
   decompressed `.out.gz`). `make_haddock_cfg.py` fixed.

`envs/redocking.yaml` needed two additions found the same way: `acpype`
(PyPI -- `biobb_chemistry` only wraps its CLI, doesn't vendor it) and
`ambertools` (conda-forge -- acpype's AM1-BCC charge step needs
antechamber/sqm on PATH, also not vendored).

**Still not run**: the actual full-sampling (`sampling=200`) pilot docking
for all 3 manifest complexes via SLURM (`run_haddock_batch.py`) --
mechanically validated at small scale, but the real-scale numbers
(HADDOCK score distributions, cluster convergence, and Stage 7's
RMSD/pocket-contact comparison against ABCfold) haven't been produced
yet. `compare_to_abcfold.py`'s `LIGAND_CHAIN_HADDOCK = "B"` assumption is
now independently confirmed correct (matches `active_passive_to_ambig`'s
default `--segid-two`), but the script itself hasn't been run against real
output.

## Repo layout

```text
redocking/
  data/
    ga1_from_ga3.sdf / .log       Stage 0 output
    manifest.csv                  Stage 2 output (this pilot's 3 complexes)
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
    run_haddock_batch.py          Stage 6 (SLURM/IFB submission)
    compare_to_abcfold.py         Stage 7
  results/
    haddock_runs/<complex_id>/    haddock3's own native run-directory output
    comparison/                   Stage 7 output (per-complex JSON + summary.csv)
```
