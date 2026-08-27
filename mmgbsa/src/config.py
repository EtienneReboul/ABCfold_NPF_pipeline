"""
mmgbsa/src/config.py
====================
Shared paths and constants for the MM-GBSA per-residue decomposition pilot
(mmgbsa/ -- see mmgbsa/README.md). Third physics-based cross-check of
ABCfold's ML co-folding of GA1 into NPF transporters, alongside redocking/
(HADDOCK3/CNS) and rescoring/ (PyRosetta REF2015).

Layout mirrors redocking/src/config.py and rescoring/src/config.py. This
project READS from redocking/ (the redocked poses + the GMM good-pose pick)
and from rescoring/ (the residue -> alignment-position map, the LDA Z-scale
loadings) and never writes to either.

Cross-project references are kept as plain file paths only -- this module is
named `config`, and so are redocking/src/config.py and rescoring/src/config.py;
Python's import cache is keyed by bare module name, so importing another
project's `config` (or any module that itself does `import config`) would
silently shadow this one. Same reasoning rescoring/src/rescore_redocked_batch.py's
module docstring spells out.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

MMGBSA_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = MMGBSA_ROOT.parent

# ---------------------------------------------------------------------------
# This project's own tree
# ---------------------------------------------------------------------------
DATA_DIR = MMGBSA_ROOT / "data"
LIGAND_PARAMS_DIR = DATA_DIR / "ligand_params"
RESULTS_DIR = MMGBSA_ROOT / "results"

SYSTEMS_DIR = RESULTS_DIR / "systems"          # Stage 2: per-complex GROMACS system
MD_DIR = RESULTS_DIR / "md"                    # Stage 3: per-(complex, replica) trajectories
MMGBSA_DIR = RESULTS_DIR / "mmgbsa"            # Stage 4-6: gmx_MMPBSA output + aggregates
FIGURES_DIR = MMGBSA_DIR / "figures"           # Stage 8
SLURM_CFG_DIR = RESULTS_DIR / "_cfgs"          # array manifests + submit scripts
SLURM_LOG_DIR = RESULTS_DIR / "_slurm_logs"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
DECOMP_BY_POSITION_CSV = MMGBSA_DIR / "decomp_by_position.csv"
BINDING_ENERGY_SUMMARY_CSV = MMGBSA_DIR / "binding_energy_summary.csv"
POSITION_SCAN_CSV = MMGBSA_DIR / "position_cohesion_scan_gbsa.csv"

# ---------------------------------------------------------------------------
# redocking/ outputs this project reads (never writes)
# ---------------------------------------------------------------------------
REDOCKING_ROOT = PIPELINE_ROOT / "redocking"
GOOD_POSE_REPRESENTATIVE_CSV = REDOCKING_ROOT / "results" / "comparison" / "good_pose_representative.csv"
HADDOCK_RUNS_DIR = REDOCKING_ROOT / "results" / "haddock_runs"
REDOCKING_MANIFEST_CSV = REDOCKING_ROOT / "data" / "manifest.csv"
GA1_SDF = REDOCKING_ROOT / "data" / "ga1_from_ga3.sdf"

# rescoring/'s own per-position Rosetta table -- Stage 7 (compare_engines.py)
# merges the GB decomposition against this on (protein, position).
ROSETTA_POSITION_ENERGETICS_CSV = REDOCKING_ROOT / "results" / "rescoring" / "position_energetics_full.csv"

# ---------------------------------------------------------------------------
# rescoring/ outputs this project reads (never writes)
# ---------------------------------------------------------------------------
RESCORING_ROOT = PIPELINE_ROOT / "rescoring"
# protein,position,resnr,is_cdd_pocket -- built by rescoring/src/build_position_mapping.py --full.
# 746 whole-alignment columns; 35 of them flagged is_cdd_pocket. NOT recomputed here.
POSITION_RESNR_MAP_FULL_CSV = RESCORING_ROOT / "data" / "position_resnr_map_full.csv"
# position, z_dim, z_name, lda_coef -- one row per (position, Z-scale); the dominant
# driver at a position is the row with the largest |lda_coef| (Stage 6 annotation).
LDA_GA1_LOADINGS_TSV = RESCORING_ROOT / "data" / "lda_GA1_loadings.tsv"

# ---------------------------------------------------------------------------
# Ligand
# ---------------------------------------------------------------------------
LIGAND_KEY = "GA1"
LIGAND_RESNAME = "GA1"           # residue name carried through to the GROMACS/Amber topology
LIGAND_NET_CHARGE = -1           # C-6 carboxylate deprotonated at physiological pH; the
                                 # C-19->C-10 lactone is neutral. Single modelled state,
                                 # documented, not scanned -- see build_ligand_params.py.

PROTEIN_CHAIN = "A"              # HADDOCK3 receptor chain (redocking/src/make_haddock_cfg.py)
LIGAND_CHAIN_HADDOCK = "B"      # HADDOCK3 ligand chain, molecule 2 -- confirmed against a real
                                 # run's log in redocking/src/compare_to_abcfold.py

# ---------------------------------------------------------------------------
# MD / MM-GBSA protocol constants (see README.md for rationale)
# ---------------------------------------------------------------------------
N_REPLICAS = 3
PROD_NS = 5.0                    # production length per replica
EQUIL_NVT_PS = 100
EQUIL_NPT_PS = 900
PROD_FRAME_EVERY_PS = 10        # -> 500 frames per 5 ns replica
GB_FRAME_INTERVAL = 5           # gmx_MMPBSA &general interval -> ~100 frames/replica scored
FORCEFIELD_PROTEIN = "amber99sb-ildn"
WATER_MODEL = "tip3p"
SALT_MOLAR = 0.150
IGB = 8                         # GBNeck2 (mbondi3 radii) -- gmx_MMPBSA sets radii to match
CA_POSRES_KJ = 100.0            # weak CA restraint during production -- no membrane, so this
                                 # keeps the transporter fold from drifting over 5 ns

# Smoke-test subset (Verification step 1): one per role.
SMOKE_COMPLEX_HINTS = ("NPF3.1_", "NPF6.1_", "NPF2.1_")


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def smoke_rows(rows: list[dict]) -> list[dict]:
    """The Verification-step-1 subset: the FIRST manifest row whose complex_id
    starts with each SMOKE_COMPLEX_HINTS prefix (one importer, one
    non_importer, one ambiguous -- NPF3.1 / NPF6.1 / NPF2.1)."""
    picked: list[dict] = []
    for hint in SMOKE_COMPLEX_HINTS:
        match = next((r for r in rows if r["complex_id"].startswith(hint)), None)
        if match is not None:
            picked.append(match)
    return picked


def final_caprieval_dir(run_dir: Path) -> Path | None:
    """redocking HADDOCK3 run dir -> its highest-numbered *_caprieval/ dir
    (the post-flexref one). Mirrors rescoring/src/rescore_redocked_batch.py's
    _find_final_caprieval_dir."""
    cands = sorted(run_dir.glob("*_caprieval"), key=lambda p: int(p.name.split("_")[0]))
    return cands[-1] if cands else None


def model_path_for_rank(run_dir: Path, rank: int) -> Path | None:
    """The flexref_*.pdb.gz path for a given caprieval_rank, from the final
    caprieval step's capri_ss.tsv. `rank` is the good_pose_representative.csv
    `caprieval_rank` (NOT always 1). Returns None if the run has no completed
    caprieval output or the model file is missing."""
    cdir = final_caprieval_dir(run_dir)
    if cdir is None:
        return None
    tsv = cdir / "capri_ss.tsv"
    if not tsv.exists():
        return None
    with tsv.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    match = next((r for r in rows if int(r["caprieval_rank"]) == int(rank)), None)
    if match is None:
        return None
    p = Path(match["model"])
    if not p.is_absolute():
        p = cdir / p
    if not p.exists():
        gz = p.with_suffix(p.suffix + ".gz")
        if gz.exists():
            return gz
        return None
    return p


def open_maybe_gzip(path: Path):
    """Text handle for a .pdb or .pdb.gz alike."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def load_position_resnr_map(csv_path: Path = POSITION_RESNR_MAP_FULL_CSV) -> dict[tuple[str, int], int]:
    """(protein, resnr) -> alignment position (1..746). Inverts
    rescoring/data/position_resnr_map_full.csv, which is keyed (protein, position)
    -> resnr. Rows with an empty resnr (protein has a gap at that column) are
    skipped."""
    out: dict[tuple[str, int], int] = {}
    for r in read_csv_rows(csv_path):
        resnr = r.get("resnr", "").strip()
        if not resnr:
            continue
        out[(r["protein"], int(float(resnr)))] = int(float(r["position"]))
    return out


def cdd_positions(csv_path: Path = POSITION_RESNR_MAP_FULL_CSV) -> set[int]:
    """The subset of alignment positions flagged is_cdd_pocket (the 35 CDD
    putative binding-site columns)."""
    return {
        int(float(r["position"]))
        for r in read_csv_rows(csv_path)
        if str(r.get("is_cdd_pocket", "")).strip().lower() in ("true", "1")
    }


def load_position_meta(csv_path: Path = POSITION_RESNR_MAP_FULL_CSV) -> dict[int, dict]:
    """alignment position (1..746) -> {is_cdd_pocket: bool, cdd_position: int|None}.
    Collapses the per-protein rows of position_resnr_map_full.csv (the flag/label
    are protein-independent)."""
    out: dict[int, dict] = {}
    for r in read_csv_rows(csv_path):
        pos = int(float(r["position"]))
        if pos in out:
            continue
        is_cdd = str(r.get("is_cdd_pocket", "")).strip().lower() in ("true", "1")
        cdd_raw = r.get("cdd_position", "").strip()
        out[pos] = {"is_cdd_pocket": is_cdd,
                    "cdd_position": int(float(cdd_raw)) if cdd_raw else None}
    return out


def load_dominant_z(tsv_path: Path = LDA_GA1_LOADINGS_TSV) -> dict[int, str]:
    """alignment position -> Z-scale name with the largest |lda_coef| at that
    position (Stage 6 annotation, matches scan_position_cohesion.py's
    dominant_z_scale)."""
    best: dict[int, tuple[float, str]] = {}
    with Path(tsv_path).open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        pos = int(float(r["position"]))
        coef = abs(float(r["lda_coef"]))
        name = r.get("z_name") or r.get("z_dim") or ""
        if pos not in best or coef > best[pos][0]:
            best[pos] = (coef, name)
    return {p: name for p, (_, name) in best.items()}


for _d in (LIGAND_PARAMS_DIR, RESULTS_DIR, MMGBSA_DIR, FIGURES_DIR, SLURM_CFG_DIR, SLURM_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
