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

# Pipeline output this project reads from (never writes to) -- the
# macro-conformation ("ca_cluster") clustering + a curated, materialized
# subset of representative CIFs per cluster, pooling BOTH apo and holo
# frames on one shared coordinate frame (worflows/postprocessing's own
# TM-alignment + PCA-k3 clustering stage, scripts/cluster_conformations.py).
# Deliberately used instead of raw results/abcfold/ -- confirmed by hand
# (2026-08-25) that a meaningful fraction of raw per-frame CIFs get
# deleted by this pipeline's own storage-compression step once run on the
# cluster (see submit_abcfold.sh's compress_abcfold_metadata.py step);
# tm_reannotated's symlinked==True frames are a separately curated,
# stable set that survives that compression.
TM_REANNOTATED_ROOT = PIPELINE_ROOT / "results" / "tm_reannotated"

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

# GA3's RCSB Chemical Component Dictionary id -- its own "model"
# coordinates (real deposited structure, currently PDB 3ED1 per the CCD
# entry's own pdbx_model_coordinates_db_code) are used directly, not the
# CCD's separate idealized conformer -- see redocking/README.md "GA1
# ligand structure" section and build_ga1_from_ga3.py.
GA3_CCD_ID = "GA3"

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


def load_importers() -> list[str]:
    """Gene names (e.g. 'NPF2.10') NPF_LDA_kernel's config.yaml flags as
    hc_importers -- the positive-control candidate pool. NOT all of these
    have a GA1-holoform ABCfold pose in THIS pipeline (some were only
    ever co-folded with a different assigned ligand here, e.g. nitrate/ABA
    -- see make_manifest.py's coverage filtering, which is what actually
    decides the usable subset, not this list alone)."""
    cfg = yaml.safe_load(NPF_LDA_KERNEL_CONFIG.read_text())
    return cfg["hc_importers"]


def load_non_importers() -> list[str]:
    """Gene names (e.g. 'NPF6.1') NPF_LDA_kernel's config.yaml flags as
    hc_non_importers -- the negative-control candidate pool. See
    load_importers()'s note -- coverage filtering still applies (a few of
    these have no CDD pocket annotation, see make_manifest.py)."""
    cfg = yaml.safe_load(NPF_LDA_KERNEL_CONFIG.read_text())
    return cfg["hc_non_importers"]


def has_cdd_residues(protein_name: str) -> bool:
    import json
    summary = json.loads(CDD_SUMMARY_JSON.read_text())
    return protein_name in summary and bool(summary[protein_name].get("residues"))


def load_cdd_residues(protein_name: str) -> list[int]:
    """CDD/InterPro putative pocket residues for one protein (structure
    numbering, no remapping needed) -- see NPF_pocket_pipeline/scripts/
    run_interproscan.py's extract_binding_site_residues(). Raises if this
    protein hasn't been CDD-annotated there yet, OR was annotated but
    InterProScan found zero domain matches for it (confirmed by hand for
    NPF4.1/NPF8.5/NPF5.9: their cached *.json has "matches": [] -- a real
    completed result, not a pending query; re-running InterProScan for
    these specific proteins would not be expected to find anything new).
    Callers that need to skip rather than fail on this should check
    has_cdd_residues() first -- see make_manifest.py."""
    import json
    summary = json.loads(CDD_SUMMARY_JSON.read_text())
    if not has_cdd_residues(protein_name):
        raise KeyError(
            f"{protein_name!r} has no CDD residues in {CDD_SUMMARY_JSON} (either not "
            f"annotated at all, or InterProScan found zero domain matches for it)."
        )
    return sorted(summary[protein_name]["residues"])


def load_cluster_assignments(protein_name: str):
    """tm_reannotated/<protein>/pca_k3/assignments.parquet -- one row per
    ABCfold frame (apo AND holo pooled), columns include status/model/
    seed/sample_index/frame_id/ptm/iptm/cluster/symlinked. Returns a
    pandas DataFrame; None if this protein has no clustering output at
    all (see make_manifest.py's coverage filtering)."""
    import pandas as pd
    path = TM_REANNOTATED_ROOT / protein_name / "pca_k3" / "assignments.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def cluster_cif_path(protein_name: str, cluster: int, frame_id: str) -> Path:
    return TM_REANNOTATED_ROOT / protein_name / "pca_k3" / f"cluster_{cluster}" / f"{frame_id}.cif"


for _d in (DATA_DIR, LIGAND_TOPOLOGY_DIR, HADDOCK_RUNS_DIR, COMPARISON_DIR):
    _d.mkdir(parents=True, exist_ok=True)
