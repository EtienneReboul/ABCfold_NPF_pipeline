# ABCfold NPF Conformation Pipeline

Predicts conformations of the Arabidopsis NPF (Nitrate/Peptide Transporter)
family with [ABCfold](https://github.com/rigdenlab/ABCFold) — AlphaFold3,
Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3 launched **together**
per protein — on the IFB Core Cluster, then clusters the resulting ensemble
by TM-helix structural alignment.

## Why not just AF3_NPF_pipeline?

`../AF3_NPF_pipeline` already samples 30 explicit AF3 seeds per protein.
`notebook/tm_conformation_clustering_gibberellin_boltz.ipynb` in that repo
overlaid those AF3 ensembles with Boltz-2 structures
(`../NPF_pocket_pipeline/results/boltz`) for the 8 Gibberellin (GA1)
importers and found AF3 alone — even with 30 seeds — does **not** recover
every conformation a second, architecturally-different model finds for the
same sequence. One model isn't enough; this pipeline runs six together
instead of resampling one.

Architecture mirrors both sibling repos: three independent stages so the
GPU-intensive model runs can happen on the cluster while pre- and
post-processing run locally.

---

## Pipeline overview

```text
┌───────────────────────────────────────────────────────────┐
│  PRE-PROCESSING (local, needs internet)                    │
│  worflows/preprocessing/                                    │
│                                                             │
│  1. Sequences    UniProt -> per-protein FASTA               │
│        v                                                    │
│  2. Fold input   fold_input.json per protein x apo/holo      │
│                  (AlphaFold3-dialect JSON, ABCfold's own      │
│                  native input format — sequence + every       │
│                  replica seed)                                │
│        v                                                    │
│  3. MMseqs2      "default run": MSA + top-hit templates from   │
│     default run  the ColabFold webserver, no manual curation,  │
│                  no pocket restraint — via ABCfold's own        │
│                  `mmseqs2msa` CLI, once per base protein,        │
│                  embedded into fold_input.resolved.json          │
└──────────────────────────┬──────────────────────────────────┘
                           │  rsync data/fold_inputs/
                           v
┌───────────────────────────────────────────────────────────┐
│  PROCESSING (IFB cluster, no internet needed)                │
│  worflows/processing/submit_abcfold.sh                        │
│                                                               │
│  4. ABCfold run — one `abcfold -abcopr ...` call per protein  │
│     x form, launching AlphaFold3 + Boltz-2 + Chai-1 +          │
│     OpenFold3 + Protenix + RosettaFold3 TOGETHER against the    │
│     same fold_input.resolved.json. Gibberellin (GA1) importers   │
│     are ordered FIRST in the SLURM array manifest — see below.    │
└──────────────────────────┬──────────────────────────────────┘
                           │  rsync results/abcfold/
                           v
┌───────────────────────────────────────────────────────────┐
│  POST-PROCESSING (local)  worflows/postprocessing/            │
│                                                               │
│  5. DeepTMHMM        TM-helix topology (BioLib)                │
│        v                                                     │
│  6. TM-helix Kabsch alignment (script), pooled across every     │
│     backend x seed — meta.csv now has a "model" column           │
│        v                                                     │
│  7. Clustering / visualisation (notebook — not yet ported, see   │
│     "Explore & cluster" below)                                    │
└───────────────────────────────────────────────────────────┘
```

## Gibberellin importers run first

`worflows/preprocessing/Snakefile` writes
`data/fold_inputs/priority_gibberellin.txt`, listing the apo/holo run
identifiers for `HC_IMPORTERS + LOW_CONFIDENCE_GA_IMPORTERS` (the GA1
ligand group: `NPF2.1`, `NPF2.5`, `NPF2.10`, `NPF2.12`, `NPF2.13`,
`NPF3.1`, `NPF4.1`, `NPF5.6`) — exactly the 8 proteins investigated in
`tm_conformation_clustering_gibberellin_boltz.ipynb`. `submit_abcfold.sh`
reads this file and puts those runs at the front of the SLURM array
manifest, so the first `--max-concurrent` array slots (and the `--test`
run) work on this group before anything else.

---

## Directory layout

```text
ABCfold_NPF_pipeline/
├── config.yaml                       ← shared defaults (tracked by git)
├── config.local.yaml.example         ← optional personal overrides (af3_sif_path override, if ever needed)
├── envs/
│   ├── pipeline.yaml                 ← controller env (install once)
│   ├── preprocessing.yaml            ← fold_input.json + mmseqs2msa resolution
│   ├── tm_analysis.yaml              ← DeepTMHMM + TM alignment
│   └── notebook.yaml                 ← clustering notebook (JupyterLab)
├── worflows/
│   ├── preprocessing/Snakefile       ← stages 1-3 (local)
│   ├── processing/submit_abcfold.sh  ← stage 4 SLURM submission (cluster)
│   └── postprocessing/Snakefile      ← stages 5-6 (local)
├── scripts/
│   ├── download_sequences.py         ← Stage 1: UniProt FASTA
│   ├── make_af3_input.py             ← Stage 2: fold_input.json + seeds
│   ├── fetch_mmseqs2_msa.py          ← Stage 3: wraps ABCfold's `mmseqs2msa`
│   ├── inject_mmseqs_msa.py          ← Stage 3: copies apo's MSA/templates onto holo
│   ├── run_deeptmhmm_topology.py     ← Stage 5: TM-helix topology
│   └── tm_helix_alignment.py         ← Stage 6: Kabsch TM alignment, multi-backend
└── data/, results/, logs/            ← created automatically (gitignored)
```

---

## Quick start

### 1. (Optional) personal overrides

```bash
cp config.local.yaml.example config.local.yaml
# nothing is required on IFB — submit_abcfold.sh auto-discovers the AF3
# Singularity image (see step 4); only override abcfold.af3_sif_path if that fails
```

### 2. Install the controller environment (once)

```bash
conda env create -f envs/pipeline.yaml
conda activate abcfold-npf-pipeline
```

### 3. Pre-processing (local, needs internet)

```bash
snakemake -s worflows/preprocessing/Snakefile --cores 4 --use-conda
```

Produces `data/fold_inputs/<protein>__{apo,holo}/fold_input.resolved.json`
for every protein x form, and `data/fold_inputs/priority_gibberellin.txt`.
This stage calls the ColabFold MMseqs2 webserver once per base protein
(`scripts/fetch_mmseqs2_msa.py`) — expect it to take a while for ~150
proteins; it's resumable (existing `fold_input.resolved.json` files are
never re-fetched).

### 4. Processing (IFB cluster)

ABCfold needs micromamba on `$PATH` (to build internal environments for
Boltz/Chai-1/OpenFold3/Protenix/RosettaFold3) and, for the AlphaFold3
backend, either Docker (`docker run`) or a Singularity/Apptainer `.sif`
image (`--af3_sif_path`). IFB compute nodes normally don't run Docker, but
**confirmed on IFB (2026-07-31): no separate image needs to be built** —
`module load alphafold/3.0.2` is itself just a thin Singularity wrapper:

```console
$ module load alphafold/3.0.2
$ which run_alphafold.py
/shared/software/singularity/wrappers/alphafold/3.0.2/run_alphafold.py
$ cat "$(which run_alphafold.py)"
#! /usr/bin/env bash
singularity exec ... 3.0.2.sif run_alphafold.py $@
```

`worflows/processing/submit_abcfold.sh`'s `discover_af3_sif()` reads that
same wrapper script and extracts its `.sif` path automatically at
submission time — you don't need to set anything. It only falls back to a
warning (and ABCfold falling back to `docker run`, which won't work on
compute nodes) if IFB ever changes that wrapper layout; in that case set
`AF3_SIF_PATH` in the script (or `config.yaml`'s `abcfold.af3_sif_path`)
manually. The script also loads the `singularity` module itself on the
compute node before calling `abcfold` (it does not load the `alphafold`
module — only its `.sif` is reused).

```bash
# Transfer inputs to the cluster
rsync -av --exclude='.snakemake/' --exclude='__pycache__/' \
  ABCfold_NPF_pipeline username@ifb-core:path/to/projects/

ssh username@ifb-core
cd path/to/projects/ABCfold_NPF_pipeline

# One-time: prime the internal micromamba backend envs (needs internet —
# run on the LOGIN node, never a compute node; downloads Boltz/Chai-1/
# OpenFold3/Protenix/RosettaFold3 weights via ABCfold's own --dry_run mode)
bash worflows/processing/submit_abcfold.sh --prime

# Dry-run first — always
bash worflows/processing/submit_abcfold.sh --dry-run

# Test one protein x form end-to-end before committing the full array
bash worflows/processing/submit_abcfold.sh --test

# Full submission
bash worflows/processing/submit_abcfold.sh
squeue -u $USER

# Transfer results back when complete
rsync -av user@ifb-core:.../results/abcfold/ results/abcfold/
```

`worflows/processing/submit_abcfold.sh`'s `abcfold` flags (`-abcopr`,
`--model_params`, `--af3_sif_path`, `--number_of_models`, `--num_recycles`,
`--no_server`, `--no_visuals`, `--override`) are confirmed against
`abcfold/argparse_utils.py` in the [ABCFold source](https://github.com/rigdenlab/ABCFold)
(main branch). What's **not** confirmed is the exact directory/file layout
ABCfold writes under `<output_dir>` per backend — the README documents the
CLI, not the output tree; `abcfold/output/*.py` suggests
`alphafold3_<name>*/`, `boltz_results_<name>*/`, `chai_output_<name>*/`,
`openfold_results_<name>*/`, `protenix_results_<name>*/`,
`rosettafold_results_<name>*/`, each holding `*.cif` somewhere under it,
but this is read from source, not a real run. Run `--test` once and inspect
`results/abcfold/<protein>/` before scaling up; adjust
`scripts/tm_helix_alignment.py`'s `BACKEND_PATTERNS` / `discover_predictions()`
if the layout doesn't match.

Unlike `AF3_NPF_pipeline`, ABCfold has no documented cache-then-reuse split
(there's no `--norun_inference`/`--norun_data_pipeline` equivalent) — one
`abcfold` call either completes or it doesn't, so a retried run always
recomputes every backend from scratch (`--override`). Only protein x form
runs with a `prediction.done` sentinel are skipped entirely.

### 5. Post-processing (local)

```bash
conda env create -f envs/tm_analysis.yaml   # once
snakemake -s worflows/postprocessing/Snakefile --cores 8 --use-conda
```

Runs DeepTMHMM (once) and the TM-helix Kabsch alignment (per protein x
form, pooled across every backend and seed), writing
`results/tm_alignment/<protein>/{aligned_ca.npy,aligned_ca_tm.npy,meta.csv,resids.csv}`.
`meta.csv` now has a `model` column (`alphafold3`/`boltz`/`chai1`/
`openfold3`/`protenix`/`rosettafold3`) alongside `seed`/`sample_index`/`ptm`.

### 6. Explore & cluster (notebook)

Not yet ported from `AF3_NPF_pipeline`. That repo's
`notebook/tm_conformation_clustering_{gibberellin,gibberellin_boltz,nitrate,other_ligand,apoform}.ipynb`
read the exact same `results/tm_alignment/<protein>/{aligned_ca_tm.npy,meta.csv}`
schema this pipeline now produces, plus one new column: color/facet by
`meta["model"]` to compare which backend(s) actually reach which cluster —
that's the whole reason this pipeline exists. `envs/notebook.yaml` here is
unchanged from that repo, so copying a notebook over and adding a
`color_by="model"` option to its plotting functions should be enough to
get started.

---

## Key configuration options (`config.yaml`)

| Key                            | Description                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| `af3.n_replicas`             | j — independent seeds per protein x form (each seed runs through all 6 backends)    |
| `mmseqs2.num_templates`      | Top-hit templates fetched per protein by`mmseqs2msa` (default run, no curation)    |
| `abcfold.models`             | Which of the 6 backends to run together (maps to`-a -b -c -o -p -r`)               |
| `abcfold.model_params`       | AlphaFold3 weights directory                                                         |
| `abcfold.af3_module_version` | Version component of IFB's`alphafold` module/wrapper path, used for auto-discovery |
| `abcfold.af3_sif_path`       | Override only — leave empty to auto-discover from the module wrapper above          |
| `abcfold.number_of_models`   | Models generated per backend per call (`--number_of_models`)                       |
| `abcfold.num_recycles`       | Recycles per backend call (`--num_recycles`; ignored by OpenFold3)                 |
| `deeptmhmm.expected_tm`      | required TM helix count per protein (12 for NPF/MFS)                                 |
| `tm_alignment.n_iter`        | Procrustes iterations for the converged mean TM structure                            |

## Resuming

- Preprocessing stages are resumable: existing output files are never
  rewritten.
- To force-rerun one protein's MMseqs2 resolution, delete its
  `data/fold_inputs/<protein>__apo/fold_input.resolved.json` (and, for
  holoform, the matching `__holo` one too — `inject_mmseqs_msa_holo` will
  re-derive it from the fresh apo resolution).
- To force-rerun one protein x form's ABCfold job, delete its
  `results/abcfold/<protein>__{apo,holo}/prediction.done` and resubmit —
  the whole `abcfold` call reruns from scratch (see the note in step 4).
- To force-rerun TM alignment for one protein x form, delete
  `results/tm_alignment/<protein>__{apo,holo}/`.
- To backfill just ONE backend for a run that already has the other 5
  (e.g. a backend bootstrap bug fixed after that run's `abcfold` call
  already completed for the rest) — **do not** call `abcfold <json>
  results/abcfold/<run> -<letter> --override` in place. Confirmed via a
  real run (RosettaFold3 backfill for `NPF2.10_Q944G5__apo`, 2026-08-06):
  `--override` doesn't scope to the backend(s) actually selected — it wipes
  the *entire* output directory first, deleting the other 5 backends'
  folders and `prediction.done` too. Recovered that one from a local rsync
  backup taken before the run; nothing was permanently lost, but the
  in-place approach must not be reused. Instead, run the missing backend
  into a disposable scratch directory, then `cp -r` only the resulting
  `<backend>_<run>/` folder into the real `results/abcfold/<run>/` —
  `worflows/processing/backfill_rosettafold3_safe.sh` does exactly this.

## Troubleshooting

| Symptom                                                                | Fix                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `abcfold: command not found`                                         | `envs/pipeline.yaml`/`envs/preprocessing.yaml` env not activated, or `python -m pip install abcfold` not yet run on the cluster node                                                                                                                                 |
| `mmseqs2msa not found on PATH`                                       | Install the`abcfold` package into `envs/preprocessing.yaml`'s env (`pip install abcfold`)                                                                                                                                                                            |
| AF3 backend fails with a Docker error on the cluster                   | `discover_af3_sif()` in `submit_abcfold.sh` failed to find IFB's `.sif` — check `/shared/software/singularity/wrappers/alphafold/$AF3_MODULE_VERSION/run_alphafold.py` still exists and still references a `.sif`, then set `AF3_SIF_PATH` manually if needed |
| `singularity: command not found` on the compute node                 | The`singularity` module failed to load — check `module avail singularity` on IFB                                                                                                                                                                                      |
| Boltz/Chai-1/OpenFold3/Protenix/RosettaFold3 fail to install mid-array | Run`bash submit_abcfold.sh --prime` on a login node (internet) first — compute nodes usually can't install anything                                                                                                                                                     |
| `tm_helix_alignment.py` finds 0 CIFs                                 | ABCfold's local output layout may not match`BACKEND_PATTERNS`/`discover_predictions()` — inspect `results/abcfold/<protein>/` and adjust                                                                                                                            |
| `No TM topology found for <protein>`                                 | DeepTMHMM predicted something other than 12 TM helices — inspect`data/interpro/deeptmhmm_TMRs.gff3`, or adjust `deeptmhmm.expected_tm`                                                                                                                                |
| Backfilling one backend in place deletes the other 5 backends' output  | `abcfold ... -<letter> --override` wipes the whole output directory, not just that backend's subfolder (confirmed 2026-08-06) — see "Resuming" above; use `backfill_rosettafold3_safe.sh`'s scratch-dir-then-merge pattern instead                                    |
| Boltz silently produces 0 predictions for a specific ligand (per-seed `manifest.json` written, no `.cif` output; log shows `found unknown escape character 'C'`) | Upstream ABCfold bug, not IFB/cluster-specific: `abcfold/boltz/af3_to_boltz.py`'s `BoltzYaml.add_key_and_value()` embeds string values (e.g. ligand SMILES) into a double-quoted YAML scalar without escaping backslashes — any SMILES with a `\` (cis/trans bond marker; **ABA** and **JA-Ile** in `config.yaml`'s `ligands:` both have one) breaks YAML parsing for every seed. Patched live in `/shared/projects/npf_abinitio/conda/envs/abcfold-npf-pipeline`'s installed copy (confirmed 2026-08-06 via a real backfill of `NPF2.14_Q9CAR9__holo`, ABA-ligand) to escape `\`/`"` first; original backed up alongside as `af3_to_boltz.py.bak-preescapefix`. Not yet upstreamed. Affects every ABA-ligand holoform run (`NPF2.14`, `NPF4.2`, `NPF4.5`, `NPF4.7`, `NPF5.1`, `NPF5.2`, `NPF5.3`, `NPF5.7`) and the one JA-Ile run (`NPF2.6`) — those hadn't reached Boltz yet as of the fix, so no other backfill should be needed for them, but worth double-checking each once its holoform run completes |
