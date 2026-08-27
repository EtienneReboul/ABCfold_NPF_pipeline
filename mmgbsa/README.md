# MM-GBSA per-residue decomposition (GA1)

Third physics-based cross-check on ABCfold's ab initio cofolding, after
`redocking/` (HADDOCK3/CNS) and `rescoring/` (PyRosetta REF2015). Where those
two score a single static pose, this one runs a **short explicit-solvent MD**
per pose and takes an **MM-GBSA endpoint with per-residue energy
decomposition** (`gmx_MMPBSA`, GB / `igb=8`), so every pocket residue gets a
sampled ligand-interaction energy — split into van der Waals, electrostatics,
polar desolvation (EGB) and non-polar desolvation (ESURF) — **with
inter-replica error bars**.

The output is mapped onto the *same* Stockholm-alignment "position" index and
scanned with the *same* Mann-Whitney U + Benjamini-Hochberg statistics as
`rescoring/src/scan_position_cohesion.py`. Headline question: **does the
MD-averaged GB decomposition independently reproduce CDD position 15** (and the
weaker secondaries 1 and 20) that the Rosetta scan flagged
(`redocking/RESULTS.md` §5)?

## Scope & key choices

| choice | value | why |
|---|---|---|
| poses | all **143** GMM good-pose representatives | `redocking/results/comparison/good_pose_representative.csv` — the full redocking corpus, so the position scan keeps its statistical power |
| membrane | **none** (solution GB) | first pass; the GA1 pocket is central/water-accessible so per-residue decomposition there is defensible. A weak Cα position-restraint (100 kJ/mol/nm²) during production keeps the transporter fold from drifting over 5 ns without a bilayer — trade-off: less backbone sampling |
| MD | ~1 ns equil (100 ps NVT + 900 ps NPT) + **5 ns production**, **3 replicas** | 500 frames/replica, GB scored on ~100 subsampled frames × 3 replicas |
| engine | GROMACS (GPU) + `gmx_MMPBSA` (CPU/MPI) | IFB has an l40s GPU partition (`worflows/processing/submit_abcfold.sh`); `gmx_MMPBSA` does per-residue decomposition natively |
| ligand | GA1 as the **−1 monoanion** (C-6 carboxylate deprotonated; C-19→C-10 lactone neutral), GAFF2 + AM1-BCC | single modelled state, not scanned |

## Setup

**Target cluster: pangloss** (`pangloss.ibmp.unistra.fr`) — it has GPU nodes
(`cryoem`: 2× L4; `gpu`: 2× Tesla) and the user's own `~/miniforge3`. IFB
works too (L40S, capped 2 GPU/user) via `--conda-root
/shared/projects/npf_abinitio/conda/envs` on the batch scripts.

**GROMACS**: checked 2026-08-27 — *neither* cluster ships a `gromacs` module
(`module spider gromacs` fails on both). Two ways to get it, and the Stage 3/4
job scripts auto-detect which is present:

1. **A site module** (preferred on pangloss, GPU build) — `mmgbsa/INSTALL.md`
   is the spec to hand to IT. The job scripts do `module load gromacs` and use
   that `gmx`.
2. **The conda-forge `gromacs`** in `envs/mmgbsa.yaml` — a portable CPU build,
   used automatically if no module is found. Fine for the smoke test; usable
   (slower) for the full run with `run_md_batch.py --cpu` on the `fast`
   partition.

```bash
# pangloss
conda env create -f envs/mmgbsa.yaml      # into ~/miniforge3/envs/mmgbsa
conda activate mmgbsa
```

`gmx_MMPBSA` is pip-only; it shells out to this env's AmberTools (`tleap`,
`parmchk2`, `cpptraj`) and to `gmx`, and needs an MPI launcher on `PATH`.

## Pipeline

```bash
cd src
python build_ligand_params.py                 # Stage 0: GA1 GAFF2 + AM1-BCC (one-time)
python make_manifest.py                        # Stage 1: 143-complex manifest from redocking/ Stage 8
python prep_systems.py                         # Stage 2: solvated GROMACS system per complex (CPU)
python run_md_batch.py --dry-run               # Stage 3: SLURM array — pangloss cryoem/L4 by default
python run_mmgbsa_batch.py --dry-run           # Stage 4: SLURM array, fast partition (429 tasks)
python aggregate_decomp.py                     # Stage 5: parse FINAL_DECOMP -> decomp_by_position.csv
python scan_positions.py                       # Stage 6: Mann-Whitney + BH scan (add --cdd-only)
python compare_engines.py                      # Stage 7: GB vs Rosetta per-position Spearman
python plots.py                                # Stage 8: static SVGs
```

Stage 3/4 default to **pangloss** (`cryoem` partition + `gpu:l4:1` for MD;
`fast` for GB; `$HOME/miniforge3/envs/mmgbsa`). Overrides:
`--cpu` (CPU-only MD on `fast`, no module needed) · `--partition/--gres` ·
`--conda-root /shared/projects/npf_abinitio/conda/envs` (IFB) ·
`--gromacs-module <name>` (`none` to force the conda `gmx`).

### Smoke test first

Every batch stage takes `--smoke` (the three `NPF3.1_` / `NPF6.1_` / `NPF2.1_`
complexes, production shortened to 100 ps, `%1` throttle). Run Stages 0–6 that
way before the full array — same "smoke test then production" discipline as
`project_redocking_pipeline_plan` in project memory.

```bash
python prep_systems.py --smoke
python run_md_batch.py --smoke               # GPU (needs the gromacs module); or add --cpu
python run_mmgbsa_batch.py --smoke
python aggregate_decomp.py --smoke
python decomp_parse.py ../results/mmgbsa/NPF3.1_*/rep0/FINAL_DECOMP_MMPBSA.dat --dump
```

If the GROMACS module isn't installed yet, run the smoke MD CPU-only:
`python run_md_batch.py --smoke --cpu` (9 × 100 ps on the `fast` partition,
minutes each).

`decomp_parse.py --dump` is the check that the `gmx_MMPBSA` decomposition file
was parsed correctly — its exact column layout is version-dependent, so
**verify it against a real smoke-test file** and tighten `decomp_parse.py` if
needed before trusting Stage 5 (same "confirm against a real run" iteration
`redocking/` went through with the CNS topology).

## Reused from sibling subrepos (never written to)

- `redocking/results/comparison/good_pose_representative.csv` — which pose per complex (Stage 1)
- `redocking/results/haddock_runs/<id>/4_flexref/*.pdb.gz` + `5_caprieval/capri_ss.tsv` — the poses
- `redocking/data/ga1_from_ga3.sdf` — GA1 3D structure (Stage 0)
- `rescoring/data/position_resnr_map_full.csv` — residue → alignment position map (Stage 5)
- `rescoring/data/lda_GA1_loadings.tsv` — dominant Z-scale per position (Stage 6)
- `redocking/results/rescoring/position_energetics_full.csv` — Rosetta per-position table (Stage 7)

## Limitations (carried into RESULTS.md)

- **No membrane.** Lipid-facing residues get spuriously favorable EGB; only the
  central pocket residues (which is all the position scan uses) are trustworthy.
- GA1 modelled as one protonation state; `gmx_MMPBSA` GB per-residue
  decomposition is approximate and **entropy is omitted** — a ranking tool, not
  a ΔG (same framing as `rescoring/src/decompose.py`'s docstring).
- Compute: ~429 GPU-h MD + 429 CPU `gmx_MMPBSA` jobs.

## Repo layout

```text
mmgbsa/
  data/
    ligand_params/     Stage 0: GA1_GMX.itp, GA1.mol2, GA1.prmtop, GA1_params.json
    manifest.csv       Stage 1
  src/
    config.py                 shared paths/constants
    build_ligand_params.py    Stage 0
    make_manifest.py          Stage 1
    prep_systems.py           Stage 2   (+ mdp_templates.py)
    run_md_batch.py           Stage 3   (+ write_prod_mdp.py, runs on the cluster)
    run_mmgbsa_batch.py       Stage 4
    aggregate_decomp.py       Stage 5   (+ decomp_parse.py)
    scan_positions.py         Stage 6
    compare_engines.py        Stage 7
    plots.py                  Stage 8
  results/
    systems/<id>/             topol.top, system.gro, index.ndx, *.mdp
    md/<id>/rep{0,1,2}/        prod.tpr, prod.xtc, prod.gro
    mmgbsa/<id>/rep{0,1,2}/    FINAL_RESULTS_MMPBSA.dat, FINAL_DECOMP_MMPBSA.csv
    mmgbsa/decomp_by_position.csv        Stage 5 (schema matches position_energetics_full.csv)
    mmgbsa/binding_energy_summary.csv    per (complex, replica) dG_GB
    mmgbsa/position_cohesion_scan_gbsa.csv  Stage 6
    mmgbsa/engine_comparison.csv         Stage 7
    mmgbsa/figures/*.svg                 Stage 8
    _cfgs/ _slurm_logs/                  SLURM array manifests + logs
```
