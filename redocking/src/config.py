"""
redocking/src/config.py
========================
Shared paths and constants for the HADDOCK3 physics-based redocking pilot
(GA1 vs. known importers/non-importers) -- see redocking/README.md.

Layout mirrors rescoring/src/config.py, but this project reads receptor
structures from this pipeline's own results/abcfold/ (both holo and apo
forms) and CDD active-residue annotations from the sibling
NPF_pocket_pipeline project, never writing to either.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REDOCKING_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REDOCKING_ROOT.parent

DATA_DIR = REDOCKING_ROOT / "data"
LIGAND_TOPOLOGY_DIR = REDOCKING_ROOT / "ligand_topology"
RESULTS_DIR = REDOCKING_ROOT / "results"
HADDOCK_RUNS_DIR = RESULTS_DIR / "haddock_runs"
COMPARISON_DIR = RESULTS_DIR / "comparison"

# Pipeline outputs this project reads from (never writes to).
ABCFOLD_OUT_ROOT = PIPELINE_ROOT / "results" / "abcfold"
LIGPOSE_ROOT = PIPELINE_ROOT / "results" / "ligand_pose"
RESCORING_MANIFEST_CSV = PIPELINE_ROOT / "rescoring" / "data" / "manifest.csv"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
GA1_FROM_GA3_SDF = DATA_DIR / "ga1_from_ga3.sdf"
GA1_FROM_GA3_LOG = DATA_DIR / "ga1_from_ga3.log"

GA1_CNS_TOP = LIGAND_TOPOLOGY_DIR / "GA1_cns.top"
GA1_CNS_PARAM = LIGAND_TOPOLOGY_DIR / "GA1_cns.param"

PROTEIN_CHAIN = "A"

# Ligand key this pilot is about -- kept as a constant (not hardcoded
# per-script) since every stage from build_ga1_from_ga3.py onward is
# GA1-specific for now.
LIGAND_KEY = "GA1"

# GA3's PDB Chemical Component Dictionary id + a real bound structure to
# pull ring-pucker-realistic coordinates from (not the CCD-idealized
# conformer) -- see redocking/README.md "GA1 ligand structure" section.
GA3_CCD_ID = "GA3"
GA3_TEMPLATE_PDB_ID = "2ZSH"  # GID1-GA3-DELLA gibberellin receptor complex

# Sibling project the CDD active-residue annotation is read from -- not a
# subdirectory of this pipeline, read directly (never written to). Same
# convention as rescoring/src/config.py's NPF_LDA_KERNEL_ROOT.
NPF_POCKET_PIPELINE_ROOT = PIPELINE_ROOT.parent / "NPF_pocket_pipeline"
CDD_SUMMARY_JSON = NPF_POCKET_PIPELINE_ROOT / "data" / "interpro" / "cdd_summary.json"

NPF_LDA_KERNEL_ROOT = PIPELINE_ROOT.parent / "NPF_LDA_kernel"
NPF_LDA_KERNEL_CONFIG = NPF_LDA_KERNEL_ROOT / "config" / "config.yaml"


def load_ligand_smiles(ligand_key: str = LIGAND_KEY) -> str:
    cfg = yaml.safe_load((PIPELINE_ROOT / "config.yaml").read_text())
    return cfg["ligands"][ligand_key]["smiles"]


def load_non_importers() -> list[str]:
    """Gene names (e.g. 'NPF6.1') NPF_LDA_kernel's config.yaml flags as
    hc_non_importers -- the negative-control receptor pool for this pilot."""
    cfg = yaml.safe_load(NPF_LDA_KERNEL_CONFIG.read_text())
    return cfg["hc_non_importers"]


def load_cdd_residues(protein_name: str) -> list[int]:
    """CDD/InterPro putative pocket residues for one protein (structure
    numbering, no remapping needed) -- see NPF_pocket_pipeline/scripts/
    run_interproscan.py's extract_binding_site_residues(). Raises if this
    protein hasn't been CDD-annotated there yet (this pilot only covers
    proteins already present in that summary; re-running InterProScan for a
    new protein is out of scope here -- do it in NPF_pocket_pipeline)."""
    import json
    summary = json.loads(CDD_SUMMARY_JSON.read_text())
    if protein_name not in summary:
        raise KeyError(
            f"{protein_name!r} not in {CDD_SUMMARY_JSON} -- run NPF_pocket_pipeline's "
            f"CDD/InterProScan stage for it first (see that project's Snakefile rule "
            f"run_interproscan)."
        )
    return sorted(summary[protein_name]["residues"])


def receptor_holo_apo_dir(protein_name: str, form: str) -> Path:
    """results/abcfold/<protein>__<form>/ -- form is 'holo' or 'apo'."""
    if form not in ("holo", "apo"):
        raise ValueError(f"form must be 'holo' or 'apo', got {form!r}")
    return ABCFOLD_OUT_ROOT / f"{protein_name}__{form}"


for _d in (DATA_DIR, LIGAND_TOPOLOGY_DIR, HADDOCK_RUNS_DIR, COMPARISON_DIR):
    _d.mkdir(parents=True, exist_ok=True)
