#!/usr/bin/env python3
"""
scripts/tm_helix_alignment.py
==============================
Stage 6 of the ABCfold NPF pipeline: automatic TM-helix structural alignment,
pooled across every backend ABCfold ran (AlphaFold3, Boltz-2, Chai-1,
OpenFold3, Protenix, RosettaFold3) and every seed.

Script version of the Kabsch/Procrustes procedure prototyped interactively
in NPF_pocket_pipeline/notebook/tm_helix_alignment.ipynb, adapted here for
ABCfold's multi-backend ensembles (forked from AF3_NPF_pipeline's
single-backend version). Kept as a plain script (not a notebook) so it can
run unattended as a Snakemake rule; notebook/tm_conformation_clustering.ipynb
is where you interact with its output.

For each protein x form, every model CIF ABCfold produced (pooled across
every backend x seed x diffusion/sample) is rigidly superposed onto a
common frame using only the Cα atoms of the 12 TM helices (Kabsch
algorithm, iterated to a converged mean TM structure). The same per-frame
rotation/translation is then applied to the entire chain, so the saved
coordinates are whole-protein alignments anchored on the transmembrane
region. Unlike the AF3-only pipeline, meta.parquet now also records which
BACKEND produced each frame — this is the whole reason ABCfold replaced
AF3_NPF_pipeline: notebook/tm_conformation_clustering_gibberellin_boltz.ipynb
showed AF3 alone doesn't recover every conformation a second model finds,
so color_by="model" in the clustering notebook is the key new axis.

TM helix boundaries come from data/interpro/tm_topology_summary.json
(parsed from data/interpro/deeptmhmm_TMRs.gff3 by
scripts/run_deeptmhmm_topology.py).

Outputs, written to results/tm_alignment/<protein>/:
  - aligned_ca.npy    — float32 (n_frames, n_ca_full, 3), whole-chain Cα, aligned
  - aligned_ca_tm.npy — float32 (n_frames, n_ca_tm, 3), TM-only Cα, aligned
  - meta.parquet      — model, seed, sample_index, ptm, iptm, rmsd_tm, rmsd_full per
                         frame, row-aligned with the two .npy above (zstd, self-
                         documenting — see scripts/parquet_utils.py)
  - resids.parquet    — resid, in_tm, tm_index residue-level TM membership, row-
                         aligned with the residue axis of aligned_ca.npy

Usage (called by Snakemake rule `tm_helix_alignment`, one protein at a time):
    python scripts/tm_helix_alignment.py \\
        --protein            NPF6.3_Q05085__apo \\
        --abcfold-output-root results/abcfold \\
        --topology           data/interpro/tm_topology_summary.json \\
        --out-root           results/tm_alignment \\
        --n-iter             5

Standalone (all proteins with a prediction.done under --abcfold-output-root):
    python scripts/tm_helix_alignment.py --abcfold-output-root results/abcfold \\
        --topology data/interpro/tm_topology_summary.json --out-root results/tm_alignment
"""

import argparse
import json
import sys
from pathlib import Path

import gemmi # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]

from abcfold_backends import (
    backend_of,
    discover_predictions,
    find_confidence,
    parse_frame_id,
)
from parquet_utils import write_parquet_with_metadata

META_TABLE_DESCRIPTION = (
    "Per-frame confidence + RMSD metadata for scripts/tm_helix_alignment.py's "
    "Kabsch-aligned Ca ensemble: one row per model CIF ABCfold produced, pooled "
    "across all 6 backends x seed x sample for one protein x form run. Row-"
    "aligned with the sibling aligned_ca.npy / aligned_ca_tm.npy in this same "
    "directory (row i here is frame i in both arrays)."
)
META_COLUMN_DESCRIPTIONS = {
    "protein": "Protein x form run identifier (e.g. NPF1.1_Q8LPL2__apo)",
    "model": "Folding backend that produced this frame: alphafold3, boltz, chai1, "
             "openfold3, protenix, or rosettafold3",
    "seed": "Random seed used for this backend run, parsed from the output path "
            "(float; NaN if unparseable)",
    "sample_index": "Sample/diffusion index within that seed, parsed from the "
                     "output path (float; NaN if unparseable, e.g. some OpenFold3 "
                     "layouts)",
    "frame_id": "Human-readable identifier combining backend/seed/sample_index for this frame",
    "ptm": "Predicted TM-score (pTM) the backend reported for this frame",
    "iptm": "Predicted interface TM-score (ipTM); NaN if the backend reports no "
            "interface (e.g. single-chain apoform) or doesn't report one",
    "rmsd_tm": "RMSD in Angstrom of this frame's 12-TM-helix Ca atoms to the "
               "converged ensemble mean, after Kabsch alignment",
    "rmsd_full": "RMSD in Angstrom of this frame's whole-chain Ca atoms to the "
                 "ensemble mean, using the same TM-helix-derived rigid transform",
}

RESIDS_TABLE_DESCRIPTION = (
    "Residue-level TM-helix membership for one protein x form run's reference "
    "frame (from data/interpro/tm_topology_summary.json). Row-aligned with the "
    "residue axis of the sibling aligned_ca.npy in this same directory."
)
RESIDS_COLUMN_DESCRIPTIONS = {
    "resid": "Residue number (mmCIF seqid) for this Ca position, from the reference frame",
    "in_tm": "True if this residue falls inside one of the 12 annotated TM helices",
    "tm_index": "0-based index of the TM helix this residue belongs to "
                "(ordered by start position), -1 if not in a TM helix",
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protein", nargs="*", default=None,
                   help="Protein name(s) to align. Default: every protein under "
                        "--abcfold-output-root with a prediction.done sentinel.")
    p.add_argument("--abcfold-output-root", default="results/abcfold")
    p.add_argument("--topology", required=True)
    p.add_argument("--out-root", default="results/tm_alignment")
    p.add_argument("--n-iter", type=int, default=5)
    return p.parse_args()


# ── TM topology ────────────────────────────────────────────────────────────────

def topology_key(protein: str) -> str:
    """Strip a '__apo'/'__holo' run suffix to recover the topology lookup
    key: TM topology is sequence-based (from DeepTMHMM) and identical for
    both forms of a protein, but topology JSON is keyed by the base protein
    name only, while `protein` here may be an apo/holo run identifier."""
    if protein.endswith("__apo") or protein.endswith("__holo"):
        return protein.rsplit("__", 1)[0]
    return protein


def tm_helices(topology: dict, protein: str) -> list[dict]:
    """Return sorted list of {start, end} dicts for the 12 TM helices."""
    entry = topology.get(topology_key(protein), {})
    for tool in ("DEEPTMHMM", "UNIPROT", "PHOBIUS", "TMHMM"):
        if entry.get(tool):
            return sorted(entry[tool], key=lambda h: h["start"])
    raise KeyError(f"No TM topology found for {protein}")


# ── ABCfold output discovery ───────────────────────────────────────────────────
# Backend detection, CIF discovery, seed/sample-id parsing and per-backend
# confidence-file lookup now live in scripts/abcfold_backends.py (shared
# with scripts/compress_abcfold_metadata.py) — imported above.


def longest_chain_name(model) -> str:
    return max(model, key=lambda c: sum(1 for _ in c)).name


def extract_ca(cif_path: Path):
    """Return (ca_coords [N,3] float32, resids [N] int32) for the longest chain."""
    structure  = gemmi.read_structure(str(cif_path))
    model      = structure[0]
    chain_name = longest_chain_name(model)
    coords, resids = [], []
    for residue in model[chain_name]:
        for atom in residue:
            if atom.name == "CA":
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                resids.append(residue.seqid.num)
    return np.array(coords, dtype=np.float32), np.array(resids, dtype=np.int32)


# ── Kabsch / iterative Procrustes ──────────────────────────────────────────────

def kabsch(P, Q):
    """Rotation R (3,3) and translation t (3,) such that (R @ P.T).T + t ~= Q."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_mean - R @ p_mean
    return R, t


def align_ensemble(tm_coords, n_iter=5, tol=1e-4):
    """Iterative Procrustes: converge tm_coords onto their own mean structure.

    Returns (ref, transforms): ref is the converged (N,3) mean TM structure,
    transforms is a list of (R, t) per frame that maps the original frame
    onto ref.
    """
    ref = tm_coords[0].copy()
    transforms = [(np.eye(3), np.zeros(3))] * len(tm_coords)
    for _ in range(n_iter):
        transforms = [kabsch(frame, ref) for frame in tm_coords]
        aligned = np.stack([(R @ frame.T).T + t
                             for frame, (R, t) in zip(tm_coords, transforms)])
        new_ref = aligned.mean(axis=0)
        shift = float(np.sqrt(((new_ref - ref) ** 2).sum(axis=1).mean()))
        ref = new_ref
        if shift < tol:
            break
    return ref, transforms


# ── Per-protein alignment ──────────────────────────────────────────────────────

def align_protein(protein: str, abcfold_output_root: Path, topology: dict,
                   out_root: Path, n_iter: int = 5) -> pd.DataFrame:
    predictions_dir = abcfold_output_root / protein
    cifs = discover_predictions(predictions_dir)
    if not cifs:
        raise FileNotFoundError(f"No CIF files found for {protein} under {predictions_dir}")

    helices = tm_helices(topology, protein)
    ref_coords, ref_resids = extract_ca(cifs[0])
    tm_mask  = np.zeros(len(ref_resids), dtype=bool)
    tm_index = np.full(len(ref_resids), -1, dtype=int)
    for i, h in enumerate(helices):
        sel = (ref_resids >= h["start"]) & (ref_resids <= h["end"])
        tm_mask |= sel
        tm_index[sel] = i
    n_ca_ref = len(ref_resids)

    full_frames, tm_frames, meta_rows = [], [], []
    by_model_count: dict[str, int] = {}
    for cif in cifs:
        coords, resids = extract_ca(cif)
        if len(coords) != n_ca_ref:
            print(f"[align] WARNING: {cif} has {len(coords)} Ca "
                  f"(expected {n_ca_ref}) - skipping")
            continue
        frame_info = parse_frame_id(cif, predictions_dir)
        full_frames.append(coords)
        tm_frames.append(coords[tm_mask])
        by_model_count[frame_info["model"]] = by_model_count.get(frame_info["model"], 0) + 1
        ptm, iptm = find_confidence(cif, frame_info["model"])
        meta_rows.append({
            "protein": protein,
            "model": frame_info["model"],
            "seed": frame_info["seed"],
            "sample_index": frame_info["sample_index"],
            "frame_id": frame_info["frame_id"],
            "ptm": ptm,
            "iptm": iptm,
        })

    n_used = len(full_frames)
    if n_used == 0:
        raise RuntimeError(f"All CIF files were skipped for {protein}")
    print(f"[align] {protein}: {n_used}/{len(cifs)} frames, "
          f"{n_ca_ref} Ca total, {int(tm_mask.sum())} in the 12 TM helices, "
          f"by model: {by_model_count}")

    ref_tm, transforms = align_ensemble(tm_frames, n_iter=n_iter)

    aligned_full, aligned_tm, rmsd_tm = [], [], []
    for full, tm, (R, t) in zip(full_frames, tm_frames, transforms):
        a_tm   = (R @ tm.T).T + t
        a_full = (R @ full.T).T + t
        aligned_tm.append(a_tm)
        aligned_full.append(a_full)
        rmsd_tm.append(float(np.sqrt(((a_tm - ref_tm) ** 2).sum(axis=1).mean())))

    ref_full_mean = np.stack(aligned_full).mean(axis=0)
    rmsd_full = [
        float(np.sqrt(((a_full - ref_full_mean) ** 2).sum(axis=1).mean()))
        for a_full in aligned_full
    ]

    for row, r_tm, r_full in zip(meta_rows, rmsd_tm, rmsd_full):
        row["rmsd_tm"], row["rmsd_full"] = round(r_tm, 4), round(r_full, 4)
    meta = pd.DataFrame(meta_rows)

    out_dir = out_root / protein
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "aligned_ca.npy", np.stack(aligned_full).astype(np.float32))
    np.save(out_dir / "aligned_ca_tm.npy", np.stack(aligned_tm).astype(np.float32))
    write_parquet_with_metadata(
        meta, out_dir / "meta.parquet",
        table_description=META_TABLE_DESCRIPTION,
        column_descriptions=META_COLUMN_DESCRIPTIONS,
    )
    write_parquet_with_metadata(
        pd.DataFrame({"resid": ref_resids, "in_tm": tm_mask, "tm_index": tm_index}),
        out_dir / "resids.parquet",
        table_description=RESIDS_TABLE_DESCRIPTION,
        column_descriptions=RESIDS_COLUMN_DESCRIPTIONS,
    )

    print(
        f"[align] {protein}: RMSD(TM) mean={meta['rmsd_tm'].mean():.3f} A, "
        f"max={meta['rmsd_tm'].max():.3f} A - saved to {out_dir}"
    )
    return meta


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    abcfold_output_root = Path(args.abcfold_output_root)
    out_root = Path(args.out_root)
    topology = json.loads(Path(args.topology).read_text())

    if args.protein:
        proteins = args.protein
    else:
        proteins = sorted(
            p.name for p in abcfold_output_root.iterdir()
            if p.is_dir() and (p / "prediction.done").exists()
        )

    if not proteins:
        raise RuntimeError(f"No proteins to align under {abcfold_output_root}")

    failures = []
    for protein in proteins:
        print(f"=== {protein} ===")
        try:
            align_protein(protein, abcfold_output_root, topology, out_root, args.n_iter)
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append(protein)

    if failures:
        print(f"\n[align] {len(failures)}/{len(proteins)} protein(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
