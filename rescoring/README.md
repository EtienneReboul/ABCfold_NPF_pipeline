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

Stages 14-16 add a second, independent method on the same manifest: ChimeraX
minimization + PLIP interaction detection, cross-checked against the CDD
putative binding site and the sequence-LDA importance (see "ChimeraX
minimization + PLIP" below).

## Setup

```bash
conda env create -f ../envs/pyrosetta_rescoring.yaml
conda activate pyrosetta_rescoring
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

PyRosetta requires an academic license (free) — the installer prompts you to
accept it. No GPU needed.

Stages 14-16 (ChimeraX + PLIP) additionally need:

- ChimeraX itself (`config.yaml`'s `chimerax_minimize.chimerax_bin`) — not a
  conda package, install separately.
- Docker running locally, with `pharmai/plip:latest` pulled.
- `pliparser`: `pip install -e /Users/ereboul/projects/python-pliparser`
  (into this same `pyrosetta_rescoring` env — pure-stdlib package, no
  additional deps).

**Running via Snakemake:** since PyRosetta's install can't be scripted into
a conda env, run `worflows/postprocessing/Snakefile`'s rescoring stages
(9-16) in a *second*, separate invocation from stages 1-8, from inside this
env and WITHOUT `--use-conda` — see the "IMPORTANT" comment right above
stage 9 in that Snakefile for the exact two commands. Also make sure the
env's own `bin/` is at the front of `PATH` (e.g. `export
PATH="$CONDA_PREFIX/bin:$PATH"` after activating) before invoking
`snakemake` directly (rather than through `conda activate` in an
interactive shell) — Snakemake's `shell:` blocks run in a subshell that
otherwise won't necessarily resolve `python` to this env.

## Pipeline

```bash
cd src
python make_manifest.py --all            # every ipTM-passing holoform frame -> data/manifest.csv
                                          # (the Snakefile's own default; omit --all for just the
                                          # capped cluster-representative sample instead)
python prep_ligand.py                    # per-ligand params/<ligand>.params (idempotent, safe to re-run)
python run_batch.py --workers 6          # resumable — skips complexes with an existing results/per_complex/<id>.csv
python build_position_mapping.py         # full-corpus pocket "position" 1-35 <-> resnr
python fit_pocket_lda.py                 # per-ligand-category sequence-LDA fit (where enough proteins exist)
python aggregate.py
python plots.py
python relax_effect_report.py            # does FastRelax actually help, by ligand/backend? see below

python chimerax_run_batch.py --workers 8            # -> results/chimerax_minimized/<id>.pdb (kept)
python plip_run_batch.py --batch-size 300 --maxthreads 8   # -> results/plip/<id>_report.txt
python plip_analysis.py                              # -> plip_cdd_agreement.csv, plip_lda_overlay.csv
```

(This mirrors `worflows/postprocessing/Snakefile`'s stages 9-13 — the
Snakefile is the primary way this runs; the above is for running a step
standalone/interactively.)

## ChimeraX minimization + PLIP (stages 14-16)

An alternative to relief.py's PyRosetta FastRelax: ChimeraX's own
`minimize` command (AMBER dock-prep charges + steepest-descent/conjugate-
gradient minimization), run on every manifest complex, then used as PLIP's
input pose. The goal isn't to replace the PyRosetta path -- it's a third,
independent line of evidence (alongside the Rosetta energetic hotspots and
the sequence-only LDA classifier) on the same question: which residues
actually matter for ligand binding? PLIP flags a residue only when it makes
an explicit, geometrically-defined interaction (H-bond, salt bridge,
hydrophobic contact, pi-stacking, ...), which is a different kind of
evidence than either Rosetta's continuous two-body energy or the LDA's
sequence-conservation signal.

**Benchmark (top-20 ipTM complexes of one protein, `NPF2.14_Q9CAR9`,
20/20 succeeded):** ChimeraX's minimizer moves the structure substantially
more than relief.py's neighborhood-restricted, coordinate-constrained
FastRelax does -- mean protein-wide heavy-atom RMSD 0.98 Å (range
0.67-1.21 Å), mean ligand heavy-atom RMSD 1.10 Å (range 0.43-1.97 Å), AMBER
energy dropping 7.6-11.6% in every case. This is expected: ChimeraX's plain
`minimize` has no interface-local restraint the way relief.py does, so it's
a genuine global relaxation of the whole model, not just clash relief.
Timing: sanitize ~0.05 s/complex (negligible), ChimeraX itself 16.6-39.2 s/
complex (mean 25.6 s/complex), serial. At 8 parallel workers this pipeline
uses, full-corpus (~16,600 complexes) wall time is on the order of
15-20 hours (some sub-linear scaling expected since each worker's AM1BCC
charge step shells out to `sqm`/`antechamber`, competing for cores) and
~11 GB of permanently-kept minimized PDBs (~694 KB/complex).

### Production stages (14-16)

```bash
cd src
python chimerax_run_batch.py --workers 8          # -> results/chimerax_minimized/<complex_id>.pdb (kept, PLIP's input)
python plip_run_batch.py --batch-size 300 --maxthreads 8   # -> results/plip/<complex_id>_report.txt
python plip_analysis.py                            # -> results/plip_cdd_agreement.csv, plip_lda_overlay.csv
```

- `chimerax_run_batch.py` — thread pool of `--workers` concurrent ChimeraX
  subprocesses (each unit of work is dominated by `subprocess.run()`, which
  releases the GIL, so threads are enough; no PyRosetta-style process-global
  registration gotcha here, so unlike `rescoring_run_complex` this isn't
  sharded per protein). Unlike PyRosetta's `staged_poses/` (deleted right
  after scoring), the minimized PDB here IS the product and is kept
  permanently — only the pre-minimization staged intermediate is scratch.
  Resumable: skips any complex_id that already has a minimized PDB.
- `plip_run_batch.py` — runs PLIP via Docker (`pharmai/plip:latest`) in
  **batches**, not one container per complex: PLIP's own `-f` flag accepts
  multiple input files plus `--maxthreads N` to process them concurrently
  inside one container invocation, confirmed by hand this avoids Docker's
  per-container startup overhead dominating wall time across 16k+ complexes
  (21 test complexes: ~3 seconds for the whole batch). PLIP auto-detects the
  ligand's HETATM records with no extra flags — no `--issmalmol`/`--chains`
  needed, confirmed on real complexes. `--nohydro` is passed since the
  minimized PDB already carries explicit hydrogens from ChimeraX's own
  dock-prep. Resumable: skips complexes that already have a report.
- `plip_analysis.py` — parses every `*_report.txt` (via the `pliparser`
  package's `plip2dictlist`, installed editable from
  `/Users/ereboul/projects/python-pliparser`), pools every receptor residue
  actually involved in a real interaction, and computes:
  - `results/plip_cdd_agreement.csv` — per protein, **precision** (of the
    residues PLIP flags as real contacts, what fraction fall inside the 35
    CDD/InterPro pocket positions `position_resnr_map.csv` defines?) and
    **recall** (of those 35 positions, how many are ever actually
    contacted?) — the direct answer to "does PLIP agree with the putative
    CDD binding site, or disagree?" Same residue-numbering space
    throughout (pose_prep.py never renumbers the protein chain, and
    decompose.py's own `prot_resi` is also the original PDB numbering, so
    no conversion is needed between Rosetta's, PLIP's, and CDD's residue
    numbers).
  - `results/plip_lda_overlay.csv` — per (ligand, position): PLIP contact
    frequency across that ligand's complexes, next to sequence-LDA
    importance (`position_importance_<ligand>.tsv`, where
    `fit_pocket_lda.py` has a fit) — mirrors `aggregate.py`'s Rosetta-vs-LDA
    overlay, giving the same comparison from PLIP's side.
  - Figures: `plip_cdd_agreement.png` (precision/recall bar per protein),
    `plip_vs_lda_scatter_<ligand>.png` for GA1/nitrate/ABA.

Requires Docker running locally and the `pliparser` package installed in
the `pyrosetta_rescoring` env (`pip install -e
/Users/ereboul/projects/python-pliparser`).

**Known accepted limitation: `dimethylarsenate` always fails in this path.**
ChimeraX's dock-prep charge step shells out to AmberTools' antechamber for
AM1-BCC charges, and antechamber's GAFF force field has no parameters for
arsenic (not a mainstream drug-discovery element, so untested/unsupported
upstream) — every dimethylarsenate complex fails with `"Failure running
ANTECHAMBER for residue ZZ6"`, unrelated to (and not fixed by) the
MM-relax step below. Not pursued further — PyRosetta rescoring (stages
9-13) still scores this ligand fine; only the ChimeraX/PLIP comparison
path is affected.

## Trying a single complex by hand

The single-complex driver these production stages were scaled up from —
useful for eyeballing one complex without running the full batch:

```bash
cd src
python run_chimerax_try.py --complex-id <id>   # e.g. any complex_id from data/manifest.csv
```

This runs two steps, each in its own file because each needs a different
Python environment:

- `sanitize_for_chimerax.py` — runs in the normal `pyrosetta_rescoring` env.
  Reuses `pose_prep.py`/`ligand_fix.py` unchanged (SMILES-corrected bond
  orders, CONECT records) to stage the pose, but *first* runs a
  ligand-geometry sanity check (bond lengths vs. covalent-radii sums, plus
  non-bonded heavy-atom clashes) and **raises instead of writing a staged
  PDB** if the pose looks broken — a raw ABCfold pose can have correct bond
  orders but still implausible bond lengths/contacts, and handing that
  straight to ChimeraX's minimizer risks a crash or garbage output. This is
  a real, targeted check (not a port of the sibling `sanitize_cif.py`'s
  generic proximity-bond-perception approach, which we don't need — this
  pipeline already has correct-by-construction bond orders from the SMILES
  template, so the only failure mode left to guard against is geometry, not
  chemistry). After that check, the ligand is also given a light RDKit MM
  relax (MMFF94, falling back to UFF -- `ligand_fix.relax_ligand_geometry`,
  in-place on the pose's own conformer, not a fresh re-embed) before being
  staged. **Fixes a real, recurring ChimeraX failure**: even a pose that
  passes the bond-length/clash check can carry enough local strain (mostly
  from `Chem.AddHs`'s heuristic hydrogen placement) that ChimeraX's own
  dock-prep valence re-perception computes an impossible electron count and
  refuses to assign AM1-BCC charges (`"<resname>: number of electrons (N) +
  formal charge (+0) is odd"`) — hit on ~4% of complexes in the first full
  batch (all ABA/ZZ3, spread across rosettafold3/openfold3/protenix
  backends). Confirmed by hand: the same 3 backend/pose combinations that
  failed before the relax step all minimize cleanly after it. Same fix that
  resolved this class of failure for the sibling Boltz-2-only pipeline this
  project generalizes from.
- `chimerax_minimize_pose.py` — runs *inside* ChimeraX itself (`chimerax
  --nogui --offscreen --script`, since `minimize`/`session` only exist in a
  live ChimeraX process), ported from the sibling project's
  `minimize_cif.py`, generalized to accept the staged PDB directly (instead
  of a raw CIF) so ChimeraX's IDATM atom-typing reads the ligand's real
  bonds from CONECT instead of re-perceiving them.

Output: `results/chimerax_try/<complex_id>_staged.pdb` and
`<complex_id>_minimized.pdb` (+ `_energy.csv` energy trajectory). Confirmed
working end-to-end on a real complex — energy converged smoothly, ligand
chemistry intact in the output.

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

**Quantified on the full 16,600-complex batch (2026-08-18) — `fa_rep`
improves, `total_score` usually doesn't, and that's not a contradiction:**
`fa_rep` drops in 99.9% of complexes (median 2230 -> 1606 REU, -28%) —
the relief step does exactly what it's designed to do, resolve steric
clashes. But `total_score` gets WORSE in 99.5% of complexes (median -7 ->
+1363 REU). Decomposing where that increase comes from: it's neither the
ligand<->protein interface (median two-body sum over every contacted
residue is ~-2 REU, negligible) nor `fa_rep` itself (which is falling, not
rising) — it's coming from elsewhere in the repacked/minimized ~10 Å
neighborhood entirely, almost certainly rotamer strain (`fa_dun`) from
repacking side chains into new rotamers, or backbone-geometry strain
(`rama_prepro`/`cart_bonded`) from the coordinate-constrained minimization
— neither is broken out by `decompose.py` (ligand<->residue two-body terms
only), so pin down further with a plain `sfxn.show(pose)` per-term
breakdown on a representative complex before/after `light_relax` if this
needs investigating further. Practical upshot: `fa_rep_raw`/
`fa_rep_relaxed` are a good before/after clash-relief signal on their own;
`total_relaxed` is not a "the pose got better" signal the way it would be
for an already-refined starting structure — the two-body
`weighted_energy`/`twobody_total` columns (what `residue_rank.csv`/
`lda_overlay.csv` actually aggregate) are scoped to the ligand interface
specifically and are far less affected by this than `total_score` is.

`relax_effect_report.py` (`results/figures/relax_effect_{overall,by_ligand,
by_backend}.png`, `results/relax_effect_summary.csv`) reproduces this as a
standing pipeline stage and breaks it out per ligand and per backend — e.g.
nitrate's small/rigid poses see the *least* fa_rep relief of any ligand
(~20% median vs. ~30-40% for the others), and protenix/rosettafold3 start
from (and stay at) the highest raw clash levels of the 6 backends. It also
breaks this down per ligand-POSE cluster (`results/figures/
relax_effect_by_pose_cluster.png`, one small panel per protein, x-axis =
that protein's own `results/ligand_pose/.../ca_cluster_<k>/cluster_<pose>`
sub-clusters — pose-cluster ids aren't comparable across proteins, so this
is deliberately faceted rather than pooled) — useful for spotting a
specific binding pose that relieves clashes noticeably worse than the
other poses found for the same protein/conformation.

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
    make_manifest.py                  complex enumeration (--all: every frame; default: cluster reps)
    run_complex.py                     single-complex CLI
    run_batch.py                        batch driver (resumable, multiprocessing)
    build_position_mapping.py            full-corpus pocket-position <-> resnr
    fit_pocket_lda.py                     per-ligand-category sequence-LDA fit
    aggregate.py                           pool + rank + LDA overlay, per ligand
    plots.py                                stacked bar / heatmap / Rosetta-vs-LDA scatter
    relax_effect_report.py                   does FastRelax help, by ligand/backend/pose cluster? (see above)
    sanitize_for_chimerax.py                  ChimeraX path: stage pose + ligand-geometry check (see above)
    chimerax_minimize_pose.py                  ChimeraX path: runs inside ChimeraX (see above)
    chimerax_run_batch.py                       ChimeraX path: production batch driver, stage 14 (see above)
    plip_run_batch.py                            PLIP path: batched docker driver, stage 15 (see above)
    plip_analysis.py                              PLIP path: CDD agreement + LDA overlay, stage 16 (see above)
    run_chimerax_try.py                            ChimeraX path: single-complex driver (see above)
  results/
    staged_poses/                per-complex corrected PDB fed to PyRosetta (deleted right after
                                  scoring -- pure scratch, regenerable from cif_path any time)
    per_complex/                  one tidy CSV per complex
    logs/                          one log per complex
    figures/                        plots.py / relax_effect_report.py / plip_analysis.py output
    all_contacts.csv, residue_rank.csv, lda_overlay.csv   aggregate.py output
    relax_effect_summary.csv                              relax_effect_report.py output
    chimerax_minimized/                                   chimerax_run_batch.py output (kept permanently -- PLIP's input)
    plip/                                                 plip_run_batch.py output (*_report.txt per complex)
    plip_contacts.csv, plip_cdd_agreement.csv, plip_lda_overlay.csv   plip_analysis.py output
    chimerax_try/                                         single-complex driver output (see above)
```
