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
region. Unlike the AF3-only pipeline, meta.csv now also records which
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
  - meta.csv          — model, seed, sample_index, ptm, iptm, rmsd_tm, rmsd_full per frame
  - resids.csv        — resid, in_tm, tm_index residue-level TM membership

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
import re
import sys
from pathlib import Path

import gemmi # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]


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
# ABCfold writes each backend's results into its own subdirectory under
# <output_dir>, named from source (per abcfold/output/*.py in the ABCFold
# source): alphafold3_<name>*, boltz_results_<name>* (or boltz_*), chai_output_<name>*
# (or chai1_*/chai_*), openfold_results_<name>* (or openfold3_*),
# protenix_results_<name>*, rosettafold_results_<name>* (or rosettafold3_*).
# This has NOT been confirmed against a real run's on-disk layout (ABCFold's
# README documents the CLI, not the output tree) — run --test first and
# inspect results/abcfold/<protein>/, then adjust BACKEND_PATTERNS below if
# a backend's folder doesn't match.

BACKEND_PATTERNS: dict[str, str] = {
    "alphafold3":   "alphafold3",
    "boltz":        "boltz",
    "chai1":        "chai",
    "openfold3":    "openfold",
    "protenix":     "protenix",
    "rosettafold3": "rosettafold",
}


def backend_of(path: Path, predictions_dir: Path) -> str:
    """Best-effort backend tag from the first path component below
    predictions_dir (e.g. 'alphafold3_NPF6.3_Q05085__apo' -> 'alphafold3')."""
    try:
        top = path.relative_to(predictions_dir).parts[0].lower()
    except (ValueError, IndexError):
        return "unknown"
    for backend, pattern in BACKEND_PATTERNS.items():
        if pattern in top:
            return backend
    return "unknown"


def discover_predictions(predictions_dir: Path) -> list[Path]:
    """Every model CIF ABCfold produced for one protein x form, pooled
    across all backend subdirectories. Templates fetched by
    scripts/fetch_mmseqs2_msa.py are cached elsewhere (data/fold_inputs/),
    never under results/abcfold/, so no template-CIF exclusion is needed
    here — but any stray 'templates' dir is skipped defensively anyway."""
    return sorted(
        c for c in predictions_dir.rglob("*.cif") if "templates" not in c.parts
    )


def parse_frame_id(cif_path: Path, predictions_dir: Path) -> dict:
    rel = cif_path.relative_to(predictions_dir)
    model = backend_of(cif_path, predictions_dir)

    m = re.search(r"seed-?(\d+)_sample-?(\d+)", str(rel), re.IGNORECASE)
    if m:
        return {"model": model, "seed": int(m.group(1)), "sample_index": int(m.group(2)),
                "frame_id": f"{model}_seed{m.group(1)}_sample{m.group(2)}"}

    m = re.search(r"seed[_-]?(\d+)", str(rel), re.IGNORECASE)
    seed = int(m.group(1)) if m else None

    m = re.search(r"model[_.]?(?:idx[_.]?)?(\d+)", cif_path.stem, re.IGNORECASE)
    sample_index = int(m.group(1)) if m else None

    frame_id = f"{model}_{rel.with_suffix('')}".replace("/", "_")
    return {"model": model, "seed": seed, "sample_index": sample_index, "frame_id": frame_id}


def _as_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _strip_model_suffix(stem: str) -> str:
    """'<base>_model' / '<base>_model_fixed' -> '<base>' (AlphaFold3,
    RosettaFold3 and OpenFold3 all write a same-named confidence file next
    to the cif with the '_model'/'_model_fixed' suffix removed)."""
    return re.sub(r"_model(_fixed)?$", "", stem)


def _confidence_af3_style(cif_path: Path) -> tuple[float, float]:
    """AlphaFold3 and RosettaFold3 write one '<base>_summary_confidences.json'
    per sample, shared by both the '_model.cif' and '_model_fixed.cif'
    variants of that sample (confirmed against a real run of each:
    results/abcfold/<protein>/{alphafold3,rosettafold}_<protein>/.../
    <base>_summary_confidences.json, top-level keys 'ptm'/'iptm').

    On a single-chain (apoform) run, AF3 writes 'iptm': null (no interface
    to score) -- correctly propagated here as NaN, not a bug. The other 5
    backends report 0.0 for the same no-interface case instead; this
    finder doesn't paper over that cross-backend inconsistency."""
    base = _strip_model_suffix(cif_path.stem)
    candidate = cif_path.with_name(f"{base}_summary_confidences.json")
    if not candidate.exists():
        return float("nan"), float("nan")
    data = json.loads(candidate.read_text())
    return _as_float(data.get("ptm")), _as_float(data.get("iptm"))


def _confidence_boltz(cif_path: Path) -> tuple[float, float]:
    """Boltz writes 'confidence_<cif_stem>.json' (prefixed, not suffixed --
    confirmed: .../predictions/<name>/confidence_<name>_model_N.json next to
    <name>_model_N.cif), top-level keys 'ptm'/'iptm'."""
    candidate = cif_path.with_name(f"confidence_{cif_path.stem}.json")
    if not candidate.exists():
        return float("nan"), float("nan")
    data = json.loads(candidate.read_text())
    return _as_float(data.get("ptm")), _as_float(data.get("iptm"))


def _confidence_chai1(cif_path: Path) -> tuple[float, float]:
    """Chai-1 stores confidence in a NumPy .npz sibling, not JSON: 'pred.
    model_idx_N.cif' pairs with 'scores.model_idx_N.npz', with flat 'ptm'/
    'iptm' float arrays (confirmed against a real run)."""
    candidate = cif_path.with_name(cif_path.name.replace("pred.", "scores.", 1)).with_suffix(".npz")
    if not candidate.exists():
        return float("nan"), float("nan")
    with np.load(candidate) as data:
        ptm = _as_float(data["ptm"][0]) if "ptm" in data.files else float("nan")
        iptm = _as_float(data["iptm"][0]) if "iptm" in data.files else float("nan")
    return ptm, iptm


def _confidence_openfold3(cif_path: Path) -> tuple[float, float]:
    """OpenFold3 writes '<base>_confidences_aggregated.json' (base = the
    sample name with the '_model'/'_model_fixed' cif suffix stripped, e.g.
    '..._sample_1_confidences_aggregated.json' next to
    '..._sample_1_model.cif'), top-level keys 'ptm'/'iptm'. The plain
    '<base>_confidences.json' sibling (also written) only has raw per-atom
    pae/plddt arrays, no scalar ptm/iptm."""
    base = _strip_model_suffix(cif_path.stem)
    candidate = cif_path.with_name(f"{base}_confidences_aggregated.json")
    if not candidate.exists():
        return float("nan"), float("nan")
    data = json.loads(candidate.read_text())
    return _as_float(data.get("ptm")), _as_float(data.get("iptm"))


def _confidence_protenix(cif_path: Path) -> tuple[float, float]:
    """Protenix writes '<base>_summary_confidence_sample_N.json' next to
    '<base>_sample_N.cif' (summary_confidence_ inserted before sample_N,
    not appended), top-level keys 'ptm'/'iptm'."""
    m = re.match(r"(.+)_sample_(\d+)$", cif_path.stem)
    if not m:
        return float("nan"), float("nan")
    base, n = m.groups()
    candidate = cif_path.with_name(f"{base}_summary_confidence_sample_{n}.json")
    if not candidate.exists():
        return float("nan"), float("nan")
    data = json.loads(candidate.read_text())
    return _as_float(data.get("ptm")), _as_float(data.get("iptm"))


CONFIDENCE_FINDERS = {
    "alphafold3":   _confidence_af3_style,
    "rosettafold3": _confidence_af3_style,
    "boltz":        _confidence_boltz,
    "chai1":        _confidence_chai1,
    "openfold3":    _confidence_openfold3,
    "protenix":     _confidence_protenix,
}

_WARNED_BACKENDS: set[str] = set()


def find_confidence(cif_path: Path, model: str) -> tuple[float, float]:
    """(ptm, iptm) for one model CIF, using the backend-specific confidence
    file layout each of ABCfold's 6 backends writes (see CONFIDENCE_FINDERS
    above -- each entry confirmed against a real run's on-disk output, not
    guessed from source). Returns (NaN, NaN) if the confidence file is
    missing or unparseable; warns at most once per backend, not once per
    frame, so a systematic layout mismatch doesn't flood stdout."""
    finder = CONFIDENCE_FINDERS.get(model)
    if finder is None:
        if model not in _WARNED_BACKENDS:
            print(f"[align] WARNING: no confidence finder for backend {model!r} - ptm/iptm will be NaN")
            _WARNED_BACKENDS.add(model)
        return float("nan"), float("nan")
    try:
        return finder(cif_path)
    except Exception as e:
        if model not in _WARNED_BACKENDS:
            print(f"[align] WARNING: confidence lookup failed for backend {model!r} ({e}) - ptm/iptm will be NaN")
            _WARNED_BACKENDS.add(model)
        return float("nan"), float("nan")


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
    meta.to_csv(out_dir / "meta.csv", index=False)
    pd.DataFrame({
        "resid": ref_resids, "in_tm": tm_mask, "tm_index": tm_index,
    }).to_csv(out_dir / "resids.csv", index=False)

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
