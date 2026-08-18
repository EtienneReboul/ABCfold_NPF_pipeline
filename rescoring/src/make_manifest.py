#!/usr/bin/env python3
"""
rescoring/src/make_manifest.py
=================================
Build data/manifest.csv: one row per complex to rescore.

Two modes:
- Default (cluster representatives): one row per protein scripts/
  cluster_conformations.py (Stage 1) has already clustered, reusing the
  representative CIFs Stage 1's `_reannotate()` already symlinked into
  results/ligand_pose/<protein>/pca_k3/ca_cluster_<k>/<tag>/cluster_<pose>/
  (capped at max_per_cluster, seeded — see cluster_conformations.py) — that
  symlinked set IS the "cluster representatives" scope this rescoring stage
  is meant to cover, so this script just enumerates it rather than
  re-deriving a different sample. For a macro-state (Ca) cluster that never
  reached ligand-pose sub-clustering (too few holo frames — see
  cluster_conformations.py's MIN_HOLO_FRAMES), falls back to that macro-
  cluster's own holo-only symlinked CIFs directly under
  results/tm_reannotated/<protein>/pca_k3/cluster_<k>/ — so no holoform
  protein/cluster is silently skipped just because it was too small for
  pose sub-clustering.
- `--all`: every holoform frame that passes the same ipTM>=0.5 filter
  `_load_protein` (scripts/cluster_conformations.py /
  scripts/_notebook_setup_functions.py) already applies before any frame is
  even eligible for clustering — i.e. the full raw ensemble those cluster
  representatives were sampled FROM, not just the sample. Resolves each
  frame's CIF directly from results/tm_alignment/<protein>__holo/meta.parquet
  (the same backend-discovery logic scripts/cluster_conformations.py uses,
  duplicated here rather than imported — that module pulls in plotly/kneed,
  not part of this project's env). complex_id naming matches the default
  mode exactly ("<protein>__holo_<frame_id>"), so a complex already scored
  via the cluster-representative manifest is recognized as done, not re-run.

Usage:
    python make_manifest.py
    python make_manifest.py --all
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

import config
import pose_prep as pp

LIGAND_POSE_TAG_RE = re.compile(r"^pca_k3_cluster(\d+)_ligandpca")

# Mirrors scripts/cluster_conformations.py's BACKEND_PATTERNS / _backend_of /
# _strip_model_suffix / _discover_abcfold_cifs / _frame_id_for_cif /
# _build_cif_by_key -- duplicated here (not imported: that module pulls in
# plotly/kneed/sklearn, not part of envs/pyrosetta_rescoring.yaml) for --all
# mode's CIF resolution. Keep in sync if that script's version changes.
BACKEND_PATTERNS = {
    "alphafold3":   "alphafold3",
    "boltz":        "boltz",
    "chai1":        "chai",
    "openfold3":    "openfold",
    "protenix":     "protenix",
    "rosettafold3": "rosettafold",
}


def _backend_of(path: Path, predictions_dir: Path) -> str:
    try:
        top = path.relative_to(predictions_dir).parts[0].lower()
    except (ValueError, IndexError):
        return "unknown"
    for backend, pattern in BACKEND_PATTERNS.items():
        if pattern in top:
            return backend
    return "unknown"


def _strip_model_suffix(stem: str) -> str:
    return re.sub(r"_model(_fixed)?$", "", stem)


def _discover_abcfold_cifs(run_name: str) -> list[Path]:
    predictions_dir = config.ABCFOLD_OUT_ROOT / run_name
    all_cifs = sorted(
        c for c in predictions_dir.rglob("*.cif") if "templates" not in c.parts
    )
    best_of_run_or_seed = f"{predictions_dir.name}_model"
    per_sample = [c for c in all_cifs if c.stem != best_of_run_or_seed]

    deduped = {}
    for c in per_sample:
        key = (c.parent, _strip_model_suffix(c.stem))
        if key not in deduped or c.stem.endswith("_fixed"):
            deduped[key] = c
    return sorted(deduped.values())


def _frame_id_for_cif(cif_path: Path, predictions_dir: Path) -> str:
    rel = cif_path.relative_to(predictions_dir)
    model = _backend_of(cif_path, predictions_dir)
    m = re.search(r"seed-?(\d+)_sample-?(\d+)", str(rel), re.IGNORECASE)
    if m:
        return f"{model}_seed{m.group(1)}_sample{m.group(2)}"
    return f"{model}_{rel.with_suffix('')}".replace("/", "_")


def _build_cif_by_key(run_name: str) -> dict[str, Path]:
    """frame_id -> resolved CIF Path, for one run (e.g. "<protein>__holo")."""
    predictions_dir = config.ABCFOLD_OUT_ROOT / run_name
    return {
        _frame_id_for_cif(c, predictions_dir): c
        for c in _discover_abcfold_cifs(run_name)
    }


def _holo_run_dir(protein: str) -> Path:
    return config.ABCFOLD_OUT_ROOT / f"{protein}__holo"


def _protein_ligand(protein: str) -> tuple[str, str]:
    """(ligand_key, ligand_chain_id) for protein's holoform run."""
    ligand_chain, smiles = pp.resolved_ligand_chain(_holo_run_dir(protein))
    ligand_key = config.ligand_key_from_smiles(smiles)
    return ligand_key, ligand_chain


def _ligand_pose_clusters(protein: str) -> dict[int, list[Path]]:
    """ca_cluster id -> list of pose-cluster directories (one per ligand
    pose, each containing symlinked representative CIFs), for whichever ca
    clusters made it far enough to get ligand-pose sub-clustering."""
    protein_dir = config.LIGPOSE_ROOT / protein / config.MACRO_METHOD_TAG
    out: dict[int, list[Path]] = {}
    if not protein_dir.exists():
        return out
    for ca_dir in sorted(protein_dir.glob("ca_cluster_*")):
        m = re.match(r"ca_cluster_(\d+)$", ca_dir.name)
        if not m:
            continue
        cid = int(m.group(1))
        # Prefer this pipeline's own tag (pca_k3_cluster<k>_ligandpca_gmm_k<n>
        # or its 1-D fallback ..._hist1d) -- ignore any other subdirectory
        # (e.g. a pre-existing "..._hdbscan_auto" or bare "..._ligandpca"
        # left over from earlier manual/exploratory notebook runs).
        tag_dirs = [
            d for d in ca_dir.iterdir()
            if d.is_dir() and LIGAND_POSE_TAG_RE.match(d.name)
            and ("_gmm_k" in d.name or d.name.endswith("_hist1d"))
        ]
        if not tag_dirs:
            continue
        pose_dirs = sorted(
            d for tag_dir in tag_dirs for d in tag_dir.glob("cluster_*") if d.is_dir()
        )
        if pose_dirs:
            out[cid] = pose_dirs
    return out


def _macro_cluster_dirs(protein: str) -> dict[int, Path]:
    protein_dir = config.REANN_ROOT / protein / config.MACRO_METHOD_TAG
    out: dict[int, Path] = {}
    if not protein_dir.exists():
        return out
    for cluster_dir in sorted(protein_dir.glob("cluster_*")):
        m = re.match(r"cluster_(\d+)$", cluster_dir.name)
        if m:
            out[int(m.group(1))] = cluster_dir
    return out


def _rows_from_cif_dir(protein: str, ligand_key: str, ca_cluster: int,
                        ligand_pose_cluster: int | None, cif_dir: Path,
                        only_holo: bool = False) -> list[dict]:
    rows = []
    for cif_symlink in sorted(cif_dir.glob("*.cif")):
        if only_holo and not cif_symlink.name.startswith("holo_"):
            continue
        try:
            cif_path = cif_symlink.resolve(strict=True)
        except FileNotFoundError:
            print(f"[make_manifest]   WARNING: broken symlink {cif_symlink}, skipping")
            continue
        unique_frame_id = cif_symlink.stem
        complex_id = f"{protein}__{unique_frame_id}"
        rows.append({
            "complex_id": complex_id,
            "protein": protein,
            "ligand": ligand_key,
            "ca_cluster": ca_cluster,
            "ligand_pose_cluster": ligand_pose_cluster if ligand_pose_cluster is not None else "",
            "cif_path": str(cif_path.relative_to(config.PIPELINE_ROOT)),
        })
    return rows


def build_manifest_for_protein(protein: str) -> list[dict]:
    if not _holo_run_dir(protein).exists():
        return []
    try:
        ligand_key, _ligand_chain = _protein_ligand(protein)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[make_manifest]   SKIP {protein}: {e}")
        return []

    pose_clusters = _ligand_pose_clusters(protein)
    macro_dirs = _macro_cluster_dirs(protein)
    if not macro_dirs:
        print(f"[make_manifest]   SKIP {protein}: no {config.MACRO_METHOD_TAG} macro-state "
              f"clustering under {config.REANN_ROOT / protein} -- run cluster_conformations.py first")
        return []

    rows: list[dict] = []
    for cid, macro_dir in macro_dirs.items():
        if cid in pose_clusters:
            for pose_dir in pose_clusters[cid]:
                pm = re.match(r"cluster_(\d+)$", pose_dir.name)
                pose_id = int(pm.group(1)) if pm else None
                rows += _rows_from_cif_dir(protein, ligand_key, cid, pose_id, pose_dir)
        else:
            # Too few holo frames for ligand-pose sub-clustering -- fall
            # back to this macro-cluster's own holo-only representatives.
            rows += _rows_from_cif_dir(protein, ligand_key, cid, None, macro_dir, only_holo=True)

    print(f"[make_manifest]   {protein} ({ligand_key}): {len(rows)} complex(es) "
          f"across {len(macro_dirs)} macro-state cluster(s), "
          f"{len(pose_clusters)} with ligand-pose sub-clusters")
    return rows


def build_all_frames_manifest_for_protein(protein: str, iptm_threshold: float = 0.5) -> list[dict]:
    """Every holoform frame for `protein` that passes ipTM>=iptm_threshold
    -- the full raw ensemble, not a cluster-representative sample. See
    module docstring."""
    if not _holo_run_dir(protein).exists():
        return []
    try:
        ligand_key, _ligand_chain = _protein_ligand(protein)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[make_manifest]   SKIP {protein}: {e}")
        return []

    meta_path = config.ALIGN_ROOT / f"{protein}__holo" / "meta.parquet"
    if not meta_path.exists():
        print(f"[make_manifest]   SKIP {protein}: no {meta_path} -- run "
              "worflows/postprocessing/Snakefile's tm_helix_alignment stage first")
        return []
    meta = pd.read_parquet(meta_path)
    n_total = len(meta)
    if iptm_threshold:
        meta = meta[meta["iptm"] >= iptm_threshold].reset_index(drop=True)

    cif_by_frame = _build_cif_by_key(f"{protein}__holo")

    # ca_cluster is informational only here (not every raw frame necessarily
    # made it into a symlinked cluster representative) -- looked up from
    # Stage 1's own assignments.parquet when available, blank otherwise.
    ca_by_frame: dict[str, int] = {}
    assign_path = config.REANN_ROOT / protein / config.MACRO_METHOD_TAG / "assignments.parquet"
    if assign_path.exists():
        assignments = pd.read_parquet(assign_path)
        ca_by_frame = dict(zip(assignments["frame_id"], assignments["cluster"]))

    rows = []
    n_missing_cif = 0
    for frame_id in meta["frame_id"]:
        cif_path = cif_by_frame.get(frame_id)
        if cif_path is None:
            n_missing_cif += 1
            continue
        unique_frame_id = f"holo_{frame_id}"
        rows.append({
            "complex_id": f"{protein}__{unique_frame_id}",
            "protein": protein,
            "ligand": ligand_key,
            "ca_cluster": ca_by_frame.get(unique_frame_id, ""),
            "ligand_pose_cluster": "",
            "cif_path": str(cif_path.relative_to(config.PIPELINE_ROOT)),
        })

    if n_missing_cif:
        print(f"[make_manifest]   {protein}: {n_missing_cif} frame(s) with no resolvable CIF, skipped")
    print(f"[make_manifest]   {protein} ({ligand_key}): {len(rows)} complex(es) "
          f"({n_total - len(meta)} dropped by ipTM < {iptm_threshold}, {n_total} raw frames total)")
    return rows


def main():
    all_frames = "--all" in sys.argv

    if not config.LIGPOSE_ROOT.exists() and not config.REANN_ROOT.exists():
        sys.exit(f"Neither {config.LIGPOSE_ROOT} nor {config.REANN_ROOT} exist -- "
                  "run scripts/cluster_conformations.py first.")

    proteins = sorted({d.name for d in config.REANN_ROOT.iterdir() if d.is_dir()}) \
        if config.REANN_ROOT.exists() else []

    all_rows: list[dict] = []
    for protein in proteins:
        if all_frames:
            all_rows += build_all_frames_manifest_for_protein(protein)
        else:
            all_rows += build_manifest_for_protein(protein)

    if not all_rows:
        sys.exit("No complexes found -- has cluster_conformations.py run for any holoform protein?")

    manifest = pd.DataFrame(all_rows).sort_values(
        ["ligand", "protein", "ca_cluster", "ligand_pose_cluster"]
    ).reset_index(drop=True)
    dupes = manifest["complex_id"].duplicated().sum()
    if dupes:
        raise RuntimeError(f"{dupes} duplicate complex_id values in manifest -- naming collision.")

    manifest.to_csv(config.MANIFEST_CSV, index=False)
    print(f"[make_manifest] wrote {config.MANIFEST_CSV} ({'all raw frames' if all_frames else 'cluster representatives'}) "
          f"({len(manifest)} complexes, {manifest.protein.nunique()} proteins, "
          f"{manifest.ligand.nunique()} ligand(s))")
    print(manifest.groupby("ligand")["complex_id"].count())


if __name__ == "__main__":
    main()
