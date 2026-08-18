# PyRosetta rescoring

Per-residue ligand↔protein REF2015 energy decomposition for cluster-
representative holoform poses, overlaid against `NPF_LDA_kernel`'s
sequence-based pocket-position importance — generalized from the sibling
`NPF_pocket_pipeline/rescoring/` project (which this was ported from; see
that project's own README/module docstrings for the original single-ligand
design this one generalizes off of).

Wired into `worflows/postprocessing/Snakefile` as stages 9-13, running
after stage 8 (`scripts/cluster_conformations.py`) — this project only
rescores the cluster-representative CIFs that stage already symlinked into
`results/tm_reannotated/`/`results/ligand_pose/`, not every ABCfold frame.

## Setup

```bash
conda env create -f ../envs/pyrosetta_rescoring.yaml
conda activate pyrosetta_rescoring
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

PyRosetta requires an academic license (free) — the installer prompts you to
accept it. No GPU needed.

**Running via Snakemake:** since PyRosetta's install can't be scripted into
a conda env, run `worflows/postprocessing/Snakefile`'s rescoring stages
(9-13) in a *second*, separate invocation from stages 1-8, from inside this
env and WITHOUT `--use-conda` — see the "IMPORTANT" comment right above
stage 9 in that Snakefile for the exact two commands.

## Pipeline

```bash
cd src
python make_manifest.py                 # cluster-representative complexes -> data/manifest.csv
python prep_ligand.py                    # per-ligand params/<ligand>.params (idempotent, safe to re-run)
python run_batch.py --workers 6          # resumable — skips complexes with an existing results/per_complex/<id>.csv
python build_position_mapping.py         # full-corpus pocket "position" 1-35 <-> resnr
python fit_pocket_lda.py                 # per-ligand-category sequence-LDA fit (where enough proteins exist)
python aggregate.py
python plots.py
```

(This mirrors `worflows/postprocessing/Snakefile`'s stages 9-13 — the
Snakefile is the primary way this runs; the above is for running a step
standalone/interactively.)

## Generalized from the sibling project — what changed and why

The sibling project rescored one ligand (GA1) from one backend's PDB output
(Boltz-2's already-minimized poses) across 33 hand-picked ("hc") proteins.
This pipeline co-folds 11 ligands across 6 backends' raw mmCIF output for
the full NPF corpus, so several things had to generalize:

- **`ligand_fix.py`**: the sibling matched ligand heavy atoms *by PDB atom
  name* against one reference pose's own CONECT-derived bond list — works
  only because Boltz-2 always names a given ligand's atoms the same way.
  Spot-checked across all 6 backends for a real complex: each backend uses
  its own, mutually inconsistent atom-naming/numbering convention *and* a
  different ligand residue name (`LIG_B`, `LIG1`, `LIG2`, `LIG0`, `l01`,
  `L:0`) — but all 6 place heavy atoms in the exact same **positional**
  (file) order, matching `Chem.MolFromSmiles(smiles)`'s own atom order.
  Confirmed for GA1 initially, then validated end-to-end (see below) for
  every ligand actually present in the manifest. So atoms are matched by
  position, not name, and each pose's corrected mol is built by copying the
  SMILES template's own already-correct bonds onto that pose's coordinates
  (no per-pose bond perception at all), with a per-pose element-sequence
  check that fails loudly (not silently) if the positional assumption ever
  doesn't hold for a given pose.
- **`pose_prep.py`**: protein atoms come from the source ABCfold CIF via
  `gemmi` (chain selection + PDB serialization) instead of PDB-text
  slicing; the ligand chain id is resolved per protein from its own
  `abc_fold_input.resolved.json` rather than assumed.
- **Params generation (`prep_ligand.py` + `ligand_fix.build_idealized_mol`)**:
  the sibling generated `LIG.params` directly from one real pose's own
  coordinates — fine for a large, floppy, low-symmetry molecule like GA1,
  but small/symmetric ligands amplify small real-pose geometric
  imperfections into an unstable Rosetta internal-coordinate (ICOOR) tree.
  Found via **nitrate** (a 4-atom, planar, resonance-symmetric ion):
  params built from a real predicted pose reliably made Rosetta's own
  pose-loading step fail (`fill_missing_atoms`) on every pose, including
  the one the params file was built from. Fix: params geometry always
  comes from an RDKit-idealized conformer (ETKDG embed + MMFF/UFF
  optimize), never a real pose — real poses are only ever used for
  per-complex scoring, via `ligand_fix.build_corrected_ligand_mol`.
- **`ADD_RING` renumbering**: found via **quercetin-3-O-sophoroside** (a
  44-heavy-atom flavonoid diglycoside — 2 aromatic rings + 3 non-aromatic
  pucker-sampled rings): `rdkit_to_params` numbers a ligand's flexible-ring
  `ADD_RING` records using RDKit's *global* ring index (including aromatic
  rings that never get their own `ADD_RING` line), so a ligand with
  aromatic and non-aromatic rings interleaved gets non-contiguous indices
  (e.g. 2, 4, 5) — Rosetta's ring-conformer-database loader indexes by that
  literal number and errors (`Cannot load database file: An invalid ring
  size was provided`) on the gaps. `prep_ligand.py` renumbers every
  `ADD_RING` index contiguously (order of appearance → 1..N) after
  `rdkit_to_params` writes the file — confirmed by hand that this exact
  change (2/4/5 → 1/2/3, nothing else touched) is what fixes the load.
- **Validation spans backends, not just proteins**: `prep_ligand.py`
  spot-checks each ligand against complexes spread across as many distinct
  backends as possible (not just distinct proteins) — the thing actually at
  risk of not generalizing (positional atom order) is a per-*backend*
  property, so validation has to exercise that axis specifically. As of
  this port, **every ligand actually present in the current dataset (ABA,
  GA1, JA-Ile, nitrate, quercetin-3-O-sophoroside) validates cleanly across
  every backend that has produced output for it** (alphafold3, boltz,
  chai1, openfold3, protenix, rosettafold3) — `prep_ligand.py` re-validates
  automatically as more ligand categories' ABCfold runs complete.
- **Manifest / pose selection**: rewritten entirely — see
  `make_manifest.py`'s docstring. Complexes are the cluster-representative
  CIFs stage 8 already symlinked (macro-state × ligand-pose cluster,
  capped and seeded there), not a fixed per-protein sample count.
- **Aggregation / overlay**: the sibling compared Rosetta energetics
  between GA1-docked importers and GA1-docked non-importers — not
  reproducible here, since holoform runs only exist for a protein's own
  assigned ligand (a nitrate transporter is never GA1-docked in this
  pipeline). `aggregate.py`/`plots.py` instead rank each ligand category's
  own Rosetta hotspots and overlay them against that category's own
  sequence-LDA importance (real published GA-importer classifier for GA1
  via `NPF_LDA_kernel`; freshly one-vs-rest-fit for categories with enough
  assigned proteins — see `fit_pocket_lda.py`) — see
  `worflows/postprocessing/Snakefile`'s stage 12 comment for which
  categories qualify.

## Raw (non-preminimized) poses — read before interpreting any single complex's score

The sibling project's poses (`model_minimized.pdb`) had already been
through an upstream minimization step before rescoring. **This pipeline's
poses are raw ABCfold co-folding output** — no minimization at all before
`relief.py`'s own light, ligand-neighborhood-restricted `FastRelax`.
Real-world consequence, confirmed on real complexes across several
backends: `fa_rep_raw` (pre-relax steric clash) commonly runs into the
thousands of REU (vs. tens-to-low-hundreds for an already-relaxed
structure), and the light relief here — deliberately restricted to a
~10 Å neighborhood, see `relief.py`'s module docstring — isn't always
enough to bring `total_relaxed` back below `total_raw` for a badly-clashing
raw pose; more relax cycles does not reliably help and can make it worse
(tested by hand: 3 cycles made one such complex's `total_relaxed` worse,
not better, on the same pose that got worse with just 1). This is the
per-pose noise the sibling README already warns about
("single-pose Rosetta energies are noisy and clash-sensitive"), just more
pronounced here because the poses start further from a physically relaxed
state — every row still carries `fa_rep_raw`/`fa_rep_relaxed`/
`total_raw`/`total_relaxed` precisely so a downstream consumer can filter
or down-weight badly-clashing raw poses rather than trust any one
complex's absolute numbers; prefer `residue_rank.csv`'s full-ensemble
aggregates.

## Sign convention & REU caveat (read before interpreting anything)

Unchanged from the sibling project:
- **Negative REU = stabilizing, positive REU = unfavorable**, everywhere.
- These are **Rosetta Energy Units, not kcal/mol** — never present them as
  binding free energies.
- Two-body ligand-residue energies omit solvation coupling beyond the
  implicit term and omit entropy entirely — this is a **triage/ranking**
  tool, not ΔG.

## Repo layout

```text
rescoring/
  data/
    manifest.csv                cluster-representative complexes to rescore
    position_resnr_map.csv      protein, position (1-35), resnr — full corpus
    lda_<ligand>_loadings.tsv   per-ligand sequence-LDA coefficients (where fitted)
  params/
    <ligand>.params             per-ligand Rosetta params (idealized-geometry, rdkit_to_params)
    <ligand>_atom_naming.json   smiles + reference complex + expected atom count/formula
  src/
    config.py                   shared paths/constants
    ligand_fix.py                positional ligand bond-order/H correction (see above)
    pose_prep.py                  CIF -> staged PDB (protein via gemmi + corrected ligand)
    prep_ligand.py                 per-ligand params generation + cross-backend validation
    relief.py                       raw score -> light coord-constrained FastRelax (unchanged from sibling)
    decompose.py                     energy-graph -> per-residue tidy table (unchanged from sibling)
    make_manifest.py                  cluster-representative complex enumeration
    run_complex.py                     single-complex CLI
    run_batch.py                        batch driver (resumable, multiprocessing)
    build_position_mapping.py            full-corpus pocket-position <-> resnr
    fit_pocket_lda.py                     per-ligand-category sequence-LDA fit
    aggregate.py                           pool + rank + LDA overlay, per ligand
    plots.py                                stacked bar / heatmap / Rosetta-vs-LDA scatter
  results/
    staged_poses/                per-complex corrected PDB fed to PyRosetta
    per_complex/                  one tidy CSV per complex
    logs/                          one log per complex
    figures/                        plots.py output
    all_contacts.csv, residue_rank.csv, lda_overlay.csv   aggregate.py output
```
