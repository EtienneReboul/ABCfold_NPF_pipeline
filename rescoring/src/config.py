"""
rescoring/src/config.py
==========================
Shared paths and constants for the PyRosetta rescoring pipeline (postprocessing
stages 9-13 — see worflows/postprocessing/Snakefile). Standalone, non-Snakemake-
internal src/ layout, mirroring the sibling NPF_pocket_pipeline/rescoring/
project this was ported and generalized from (see that project's README.md
for the original design rationale — bond-order-fix the ligand, light
neighborhood-restricted FastRelax, per-residue REF2015 decomposition).

Generalized from that single-ligand (GA1), single-backend (Boltz-2 PDB),
33-protein port to this pipeline's reality: 11 co-folded ligands (config.yaml's
`ligands:` dict), 6 backends' mmCIF output, and the full protein corpus.
"""
from __future__ import annotations

from pathlib import Path

import yaml

RESCORING_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = RESCORING_ROOT.parent

DATA_DIR = RESCORING_ROOT / "data"
PARAMS_DIR = RESCORING_ROOT / "params"
RESULTS_DIR = RESCORING_ROOT / "results"
STAGED_DIR = RESULTS_DIR / "staged_poses"
PER_COMPLEX_DIR = RESULTS_DIR / "per_complex"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"

# Pipeline outputs this project reads from (never writes to).
ABCFOLD_OUT_ROOT = PIPELINE_ROOT / "results" / "abcfold"
ALIGN_ROOT = PIPELINE_ROOT / "results" / "tm_alignment"
REANN_ROOT = PIPELINE_ROOT / "results" / "tm_reannotated"
LIGPOSE_ROOT = PIPELINE_ROOT / "results" / "ligand_pose"
SEQUENCES_DIR = PIPELINE_ROOT / "data" / "sequences"
SEQ_SENTINEL = SEQUENCES_DIR / "sequences.done"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
POSITION_RESNR_MAP_CSV = DATA_DIR / "position_resnr_map.csv"
# Whole-alignment (all 746 npf_aligned.sto columns, not just the 35 CDD
# pocket ones) version of the same mapping -- see
# build_position_mapping.py's --full mode. 2026-08-26, at the user's
# request: a Rosetta-contacted residue outside the CDD-annotated pocket
# is still a real signal the 35-position-only scan silently drops.
POSITION_RESNR_MAP_FULL_CSV = DATA_DIR / "position_resnr_map_full.csv"

# The clustering method_tag every complex in the manifest is drawn from —
# see scripts/cluster_conformations.py's MACRO_METHOD_TAG. Fixed here (not
# discovered per-protein) since that script always uses this one tag for
# the macro-state GMM k=3 pass every ligand-pose cluster builds on.
MACRO_METHOD_TAG = "pca_k3"

# Staged-PDB convention: the ligand's own chain id ("L") and the protein
# chain ("A") are shared/fixed across every ligand — but NOT the residue
# name, unlike the sibling project's single-ligand "LIG" convention. Why:
# PyRosetta registers each `-extra_res_fa` residue TYPE by its NAME into
# one process-wide ResidueTypeSet, and `pyrosetta.init()`'s core run-level
# flags only take effect on the first call in a process — but a SECOND
# `-extra_res_fa` call for a DIFFERENT ligand sharing the same residue name
# is silently ignored (confirmed by hand: two different ligands both named
# "LIG", loaded via two separate pyrosetta.init() calls in one process,
# resulted in the SECOND ligand's real poses being scored using the FIRST
# ligand's residue type -- wrong atom count, wrong chemistry, no error
# raised). run_batch.py's worker processes handle complexes across
# MULTIPLE ligands (manifest rows aren't grouped by ligand before being
# queued to workers), so every ligand needs its own distinct residue name
# for a second `-extra_res_fa` load to actually register additively
# (confirmed: DISTINCT names load correctly in the same process) instead of
# silently reusing whatever was loaded first.
# Deliberately synthetic-looking (ZZ-prefixed) codes, not "real-looking"
# 3-letter guesses like "ABA"/"GA1" -- Rosetta ships its own base residue
# type set (residue_types.txt, real PDB chemical-component codes among
# others) and errors loudly ("residue type 'X' already exists in the
# cache") if a custom -extra_res_fa name collides with one already in it.
# Confirmed the hard way: "ABA" collides with Rosetta's own built-in
# noncanonical-amino-acid entry for alpha-aminobutyric acid. Every ZZ1..ZZA
# candidate below was spot-checked empirically (fresh process, trivial
# single-atom mol) and loads with no collision.
LIGAND_CODES = {
    "GA1": "ZZ1",
    "nitrate": "ZZ2",
    "ABA": "ZZ3",
    "auxin": "ZZ4",
    "glycerate": "ZZ5",
    "dimethylarsenate": "ZZ6",
    "glycylglycine": "ZZ7",
    "quercetin-3-O-sophoroside": "ZZ8",
    "spermidine": "ZZ9",
    "JA-Ile": "ZZA",
}
LIGAND_CHAIN = "L"
PROTEIN_CHAIN = "A"


def ligand_resname(ligand_key: str) -> str:
    if ligand_key not in LIGAND_CODES:
        raise KeyError(f"{ligand_key!r} has no entry in LIGAND_CODES -- add a short, "
                        f"unique (<=3-4 char) residue code for it. Known: {sorted(LIGAND_CODES)}")
    return LIGAND_CODES[ligand_key]

# The CDD Feature-1 (cd17351) MSA anchor: the one root-entry sequence in the
# NPF corpus, whose own binding-site residues define the 35 alignment
# columns used as pocket "position" 1..35 everywhere in NPF_LDA_kernel's
# outputs (and this project's position_resnr_map.csv). Same anchor the
# sibling rescoring project used.
ANCHOR_PROTEIN = "NPF6.1_Q9LYR6"
ANCHOR_UNIPROT_ID = "Q9LYR6"

# Sibling project this pocket-position framework and LDA-fitting method are
# reused from — not a subdirectory of this pipeline, read directly (never
# written to).
NPF_LDA_KERNEL_ROOT = PIPELINE_ROOT.parent / "NPF_LDA_kernel"
NPF_LDA_KERNEL_ALIGNMENT = NPF_LDA_KERNEL_ROOT / "data" / "cdd_msa" / "npf_aligned.sto"
NPF_LDA_KERNEL_ANCHOR_BINDING_SITE = (
    NPF_LDA_KERNEL_ROOT / "data" / "interpro" / f"{ANCHOR_PROTEIN}_binding_site_residues.txt"
)
NPF_LDA_KERNEL_POCKET_SITES = NPF_LDA_KERNEL_ROOT / "results" / "ga_classifier" / "pocket_sites_cdd_msa.tsv"
NPF_LDA_KERNEL_GA_LABELS = NPF_LDA_KERNEL_ROOT / "results" / "ga_classifier" / "labels.tsv"


def load_ligands_config() -> dict:
    """config.yaml's `ligands:` dict -- {ligand_key: {"smiles": ...}}."""
    cfg = yaml.safe_load((PIPELINE_ROOT / "config.yaml").read_text())
    return cfg["ligands"]


def load_ligand_smiles(ligand_key: str) -> str:
    ligands = load_ligands_config()
    if ligand_key not in ligands:
        raise KeyError(f"{ligand_key!r} not found in config.yaml's ligands: "
                        f"{sorted(ligands)}")
    return ligands[ligand_key]["smiles"]


def ligand_key_from_smiles(smiles: str) -> str:
    """Reverse lookup: which config.yaml ligand key produced this SMILES
    (as read from one complex's own abc_fold_input.resolved.json) -- avoids
    needing worflows/preprocessing/Snakefile's ligand_for() protein lists
    here, since the resolved SMILES is already authoritative per-complex."""
    for key, entry in load_ligands_config().items():
        if entry["smiles"] == smiles:
            return key
    raise ValueError(f"No config.yaml ligand key has smiles={smiles!r}")


def params_path(ligand_key: str) -> Path:
    return PARAMS_DIR / f"{ligand_key}.params"


def all_params_paths() -> list[Path]:
    """Every ligand's params/<ligand>.params currently on disk (i.e. every
    ligand prep_ligand.py has successfully validated) -- PyRosetta only
    registers custom `-extra_res_fa` residue types passed in its FIRST
    `pyrosetta.init()` call in a process; a second call naming a different
    params file is silently ignored (confirmed by hand: PyRosetta falls
    back to auto-perceiving a generic "pdb_<resname>" residue from raw
    coordinates instead -- no error, just quietly wrong bond orders/
    hydrogens, exactly what ligand_fix.py exists to prevent). So every
    worker process that might score more than one ligand (run_batch.py's
    workers do, since manifest rows aren't grouped by ligand before being
    queued) must load every ligand's params in ONE combined init call,
    before scoring anything -- see relief.py's init_pyrosetta."""
    return sorted(p for p in PARAMS_DIR.glob("*.params") if not p.name.startswith("_"))


def atom_naming_path(ligand_key: str) -> Path:
    return PARAMS_DIR / f"{ligand_key}_atom_naming.json"


for _d in (DATA_DIR, PARAMS_DIR, STAGED_DIR, PER_COMPLEX_DIR, LOGS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
