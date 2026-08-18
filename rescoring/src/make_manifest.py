#!/usr/bin/env python3
"""
rescoring/src/make_manifest.py
=================================
Build data/manifest.csv: one row per cluster-representative complex to
rescore, for every protein scripts/cluster_conformations.py (Stage 1) has
already clustered.

Rather than re-sampling frames itself, this reuses the representative CIFs
Stage 1's `_reannotate()` already symlinked into
results/ligand_pose/<protein>/pca_k3/ca_cluster_<k>/<tag>/cluster_<pose>/
(capped at max_per_cluster, seeded — see cluster_conformations.py) — that
symlinked set IS the "cluster representatives" scope this rescoring stage
is meant to cover, so this script just enumerates it rather than
re-deriving a different sample.

For a macro-state (Ca) cluster that never reached ligand-pose sub-
clustering (too few holo frames — see cluster_conformations.py's
MIN_HOLO_FRAMES), falls back to that macro-cluster's own holo-only
symlinked CIFs directly under
results/tm_reannotated/<protein>/pca_k3/cluster_<k>/ (filenames are
"holo_..." / "apo_..." prefixed by unique_frame_id's own status prefix —
see cluster_conformations.py's _reannotate calls) — so no holoform
protein/cluster is silently skipped just because it was too small for
pose sub-clustering.

Usage:
    python make_manifest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

import config
import pose_prep as pp

LIGAND_POSE_TAG_RE = re.compile(r"^pca_k3_cluster(\d+)_ligandpca")


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


def main():
    if not config.LIGPOSE_ROOT.exists() and not config.REANN_ROOT.exists():
        sys.exit(f"Neither {config.LIGPOSE_ROOT} nor {config.REANN_ROOT} exist -- "
                  "run scripts/cluster_conformations.py first.")

    proteins = sorted({d.name for d in config.REANN_ROOT.iterdir() if d.is_dir()}) \
        if config.REANN_ROOT.exists() else []

    all_rows: list[dict] = []
    for protein in proteins:
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
    print(f"[make_manifest] wrote {config.MANIFEST_CSV} "
          f"({len(manifest)} complexes, {manifest.protein.nunique()} proteins, "
          f"{manifest.ligand.nunique()} ligand(s))")
    print(manifest.groupby("ligand")["complex_id"].count())


if __name__ == "__main__":
    main()
