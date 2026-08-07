#!/usr/bin/env python3
"""
scripts/compress_abcfold_metadata.py
=======================================
Stage 7 of the ABCfold NPF pipeline: convert ABCfold's raw, per-sample
confidence sprawl (one 12-17MB pretty-printed JSON/npz/npy per sample,
across 6 backends x ~10 seeds x 5 samples per protein x form) into two
compact artifacts per run, mirroring results/tm_alignment/<protein>/:

  results/metadata/<protein__form>/
    model_metadata.parquet   - one row per (backend, seed, sample_index):
                                ptm, iptm, ranking_score, fraction_disordered,
                                has_clash, mean_plddt + a lossless backend-
                                specific extra_json column. zstd-compressed.
    arrays.h5                - pae / contact_probs / atom_plddt per sample,
                                float16, gzip+shuffle compressed, keyed by
                                each cif's own path (relative to the run's
                                predictions dir, extension stripped) so the
                                mapping back to results/abcfold/ is direct.
  results/metadata/all_models.parquet
                            - concat of every run's model_metadata.parquet

All 6 backends normalize their full per-sample confidence data onto the
same AF3 schema (confirmed by direct inspection of a completed run):
'pae' (NxN), 'contact_probs' (NxN), 'atom_plddts' (per-atom), plus
'token_res_ids'/'token_chain_ids'/'atom_chain_ids' (topology, identical
across every sample from the same backend x protein - written once per
backend, not duplicated per sample). Chai-1 is the one exception: it keeps
all 5 models' PAE for a seed in one real (non-JSON) float32 .npy, with no
separate contact_probs/atom_plddts array (Chai's per-atom pLDDT lives in
the CIF's own B-factor column instead, already preserved by keeping the
CIF).

Precision: pae is capped at 31.75 A and already bucketed at inference time,
contact_probs is a [0,1] probability, atom_plddts is a [0,100] confidence -
float16 keeps ~0.02 A / ~0.0005 / ~0.06 resolution respectively, well under
each field's own native granularity. Scalars (ptm/iptm/ranking_score/...)
stay float32 - one value per row, quantizing them further saves nothing.

Safety: default mode is WRITE-ONLY. The untouched results/abcfold/<run>/
tree is left exactly as-is; this only adds results/metadata/<run>/ next to
it. Every write is verified (row count against CIFs discovered, array
shapes, a handful of decoded float16 values compared back against source)
before anything is considered clean. Pass --delete-originals to also
remove, ONLY for runs whose verification passed, the raw files just
replaced:
  - the big per-sample confidence files this script reads (AF3/RosettaFold3/
    OpenFold3's '*_confidences.json', Boltz's 'pae_*.npz', Protenix's
    '*_full_data_sample_N.json', Chai's 'pae_scores.npy')
  - Boltz's 'pde_*.npz'/'plddt_*.npz' (backend-native extras outside the
    unified schema above - pde has no downstream consumer, plddt is
    redundant with atom_plddts already inside pae_*.npz)
  - AF3's per-seed 'seed-*_distogram/*.npz' (zero consumers anywhere)
  - Boltz's 'processed/' (its own input-featurization cache, not a result)
    and 'lightning_logs/'
  - Chai's 'templates/*.cif.gz' (input templates) and redundant
    'pae_scores_model_N.npy' copies (see generate_abcfold_visuals.py's
    clean_chai_pae_copies() docstring)
  - '.DS_Store' files, AF3's top-level '<name>_data.json' (redundant with
    'abc_fold_input.resolved.json')

Usage:
    # write-only, one run
    python scripts/compress_abcfold_metadata.py \\
        --protein NPF1.1_Q8LPL2__apo --abcfold-output-root results/abcfold \\
        --out-root results/metadata

    # every run with a prediction.done sentinel, write-only
    python scripts/compress_abcfold_metadata.py \\
        --abcfold-output-root results/abcfold --out-root results/metadata

    # after inspecting results/metadata/<run>/, reclaim the disk space
    python scripts/compress_abcfold_metadata.py \\
        --protein NPF1.1_Q8LPL2__apo --delete-originals
"""

import argparse
import json
import re
import sys
from pathlib import Path

import h5py  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]

from abcfold_backends import (
    RAW_CONFIDENCE_PATH_FINDERS,
    SUMMARY_PATH_FINDERS,
    discover_predictions,
    parse_frame_id,
)
from parquet_utils import write_parquet_with_metadata

ARRAY_KEYS = ("pae", "contact_probs", "atom_plddt")
SCALAR_COLUMNS = ("ptm", "iptm", "ranking_score", "fraction_disordered", "mean_plddt")

MODEL_METADATA_TABLE_DESCRIPTION = (
    "Standardized per-sample scalar confidence metadata for one ABCfold run "
    "(protein x form), replacing the raw per-sample confidence JSON/npz/npy "
    "sprawl under results/abcfold/<run>/ (see this script's module docstring). "
    "One row per model CIF pooled across all 6 backends x seed x sample. The "
    "full pae/contact_probs/atom_plddt arrays for a row (when has_array is "
    "True) live in this run's sibling arrays.h5, keyed by this row's "
    "array_key. Generated by scripts/compress_abcfold_metadata.py."
)
MODEL_METADATA_COLUMN_DESCRIPTIONS = {
    "protein": "Base protein identifier (without the __apo/__holo suffix)",
    "form": "'apo' or 'holo' -- whether this run co-folded a ligand",
    "backend": "Folding backend that produced this sample: alphafold3, boltz, "
               "chai1, openfold3, protenix, or rosettafold3",
    "seed": "Random seed used for this backend run, best-effort parsed from the "
            "output path (may be null)",
    "sample_index": "Sample/diffusion/model index within that seed, best-effort "
                     "parsed from the output path (may be null, e.g. some "
                     "OpenFold3 path layouts)",
    "frame_id": "Human-readable identifier combining backend/seed/sample_index",
    "cif_path": "Path to this sample's structure file, relative to --abcfold-output-root",
    "array_key": "This sample's group key under the run's arrays.h5 (holds "
                 "pae/contact_probs/atom_plddt when has_array is True); also "
                 "equal to cif_path with its run prefix and .cif extension removed",
    "extra_json": "The full raw scalar-confidence dict this backend wrote for "
                  "this sample (JSON text) -- the lossless source the ptm/iptm/"
                  "ranking_score/fraction_disordered/mean_plddt/has_clash "
                  "columns below were mapped from",
    "has_array": "True if pae/contact_probs/atom_plddt for this sample were "
                 "written to the sibling arrays.h5",
    "ptm": "Predicted TM-score (pTM) reported by the backend for this sample",
    "iptm": "Predicted interface TM-score (ipTM); null/0 depending on backend "
            "convention when there is no interface to score",
    "ranking_score": "Backend's own model-ranking score (source field name "
                      "varies by backend: ranking_score/confidence_score/"
                      "aggregate_score/sample_ranking_score)",
    "fraction_disordered": "Fraction of residues the backend flags as "
                            "disordered (source field varies: "
                            "fraction_disordered/disorder); null if not reported",
    "mean_plddt": "Mean predicted LDDT (0-100 scale) if this backend reports a "
                  "single scalar summary value; null otherwise (AlphaFold3, "
                  "RosettaFold3 and Chai-1 don't)",
    "has_clash": "Backend's own clash flag/score for this sample; null if not reported",
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protein", nargs="*", default=None,
                    help="Run name(s) to convert (e.g. NPF1.1_Q8LPL2__apo). Default: "
                         "every run under --abcfold-output-root with a prediction.done sentinel.")
    p.add_argument("--abcfold-output-root", default="results/abcfold")
    p.add_argument("--out-root", default="results/metadata")
    p.add_argument("--sample-checks", type=int, default=5,
                    help="Number of arrays.h5 entries to spot-check against source per run")
    p.add_argument("--delete-originals", action="store_true",
                    help="After verification passes for a run, delete the raw files it replaced. "
                         "Default is write-only (originals untouched).")
    p.add_argument("--skip-merge", action="store_true",
                    help="Skip rebuilding results/metadata/all_models.parquet at the end")
    return p.parse_args()


# ── Scalar summary-file parsing ─────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_summary(backend: str, cif_path: Path) -> dict | None:
    finder = SUMMARY_PATH_FINDERS.get(backend)
    if finder is None:
        return None
    path = finder(cif_path)
    if path is None or not path.exists():
        return None
    if backend == "chai1":
        with np.load(path) as data:
            return {k: data[k].tolist() for k in data.files}
    return _load_json(path)


def unify_scalar_row(backend: str, raw: dict) -> dict:
    """Map each backend's own summary-confidence schema onto one common set
    of columns for convenient cross-backend querying. `raw` (kept verbatim
    in extra_json) remains the source of truth for anything backend-specific
    this mapping drops (e.g. per-chain breakdowns)."""

    def g(*keys, default=None):
        for k in keys:
            if k in raw:
                v = raw[k]
                if isinstance(v, list):
                    v = v[0] if v else default
                return v
        return default

    if backend in ("alphafold3", "rosettafold3"):
        return dict(
            ptm=g("ptm"), iptm=g("iptm"), ranking_score=g("ranking_score"),
            fraction_disordered=g("fraction_disordered"), has_clash=g("has_clash"),
            mean_plddt=None,
        )
    if backend == "boltz":
        plddt = g("complex_plddt")
        return dict(
            ptm=g("ptm"), iptm=g("iptm"), ranking_score=g("confidence_score"),
            fraction_disordered=None, has_clash=None,
            mean_plddt=(plddt * 100.0) if plddt is not None else None,
        )
    if backend == "chai1":
        return dict(
            ptm=g("ptm"), iptm=g("iptm"), ranking_score=g("aggregate_score"),
            fraction_disordered=None, has_clash=g("has_inter_chain_clashes"),
            mean_plddt=None,
        )
    if backend == "openfold3":
        return dict(
            ptm=g("ptm"), iptm=g("iptm"), ranking_score=g("sample_ranking_score"),
            fraction_disordered=g("disorder"), has_clash=g("has_clash"),
            mean_plddt=g("avg_plddt"),
        )
    if backend == "protenix":
        return dict(
            ptm=g("ptm"), iptm=g("iptm"), ranking_score=g("ranking_score"),
            fraction_disordered=g("disorder"), has_clash=g("has_clash"),
            mean_plddt=g("plddt"),
        )
    return dict(ptm=None, iptm=None, ranking_score=None, fraction_disordered=None,
                has_clash=None, mean_plddt=None)


# ── Large array-file parsing ─────────────────────────────────────────────────

def raw_array_path(backend: str, cif_path: Path) -> Path | None:
    finder = RAW_CONFIDENCE_PATH_FINDERS.get(backend)
    if finder is None:
        return None
    return finder(cif_path)


def load_array_payload(backend: str, cif_path: Path, raw_path: Path):
    """(arrays, topology) for one sample: arrays has whichever of
    pae/contact_probs/atom_plddt this backend provides (float16); topology
    has token_res_ids/token_chain_ids/atom_chain_ids (int32/bytes), written
    by the caller only once per backend since it's identical across every
    sample a given backend produces for this protein."""
    if not raw_path.exists():
        return {}, {}

    if backend == "chai1":
        m = re.search(r"model_idx_(\d+)", cif_path.name)
        if not m:
            return {}, {}
        idx = int(m.group(1))
        arr = np.load(raw_path)
        if idx >= arr.shape[0]:
            return {}, {}
        return {"pae": arr[idx].astype(np.float16)}, {}

    data = _load_json(raw_path)
    arrays = {}
    if "pae" in data:
        arrays["pae"] = np.asarray(data["pae"], dtype=np.float16)
    if "contact_probs" in data:
        arrays["contact_probs"] = np.asarray(data["contact_probs"], dtype=np.float16)
    if "atom_plddts" in data:
        arrays["atom_plddt"] = np.asarray(data["atom_plddts"], dtype=np.float16)

    topology = {}
    if "token_res_ids" in data:
        topology["token_res_ids"] = np.asarray(data["token_res_ids"], dtype=np.int32)
    if "token_chain_ids" in data:
        topology["token_chain_ids"] = np.asarray(data["token_chain_ids"], dtype="S1")
    if "atom_chain_ids" in data:
        topology["atom_chain_ids"] = np.asarray(data["atom_chain_ids"], dtype="S1")
    return arrays, topology


# ── Per-run conversion ───────────────────────────────────────────────────────

def run_to_protein_form(run: str) -> tuple[str, str]:
    if run.endswith("__apo") or run.endswith("__holo"):
        protein, form = run.rsplit("__", 1)
        return protein, form
    return run, ""


def process_run(run: str, abcfold_output_root: Path, out_root: Path) -> dict:
    predictions_dir = abcfold_output_root / run
    cifs = discover_predictions(predictions_dir)
    if not cifs:
        raise FileNotFoundError(f"No CIF files found for {run} under {predictions_dir}")

    protein, form = run_to_protein_form(run)
    out_dir = out_root / run
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "arrays.h5"
    parquet_path = out_dir / "model_metadata.parquet"

    rows = []
    topology_written: set[str] = set()
    raw_files_seen: set[Path] = set()
    bytes_before = 0
    array_datasets = 0

    with h5py.File(h5_path, "w") as h5f:
        for cif in cifs:
            frame = parse_frame_id(cif, predictions_dir)
            backend = frame["model"]
            array_key = cif.relative_to(predictions_dir).with_suffix("").as_posix()

            raw_summary = load_summary(backend, cif)
            unified = unify_scalar_row(backend, raw_summary) if raw_summary else {}

            row = {
                "protein": protein, "form": form, "backend": backend,
                "seed": frame["seed"], "sample_index": frame["sample_index"],
                "frame_id": frame["frame_id"],
                "cif_path": cif.relative_to(abcfold_output_root).as_posix(),
                "array_key": array_key,
                "extra_json": json.dumps(raw_summary) if raw_summary else None,
                "has_array": False,
                **{k: unified.get(k) for k in SCALAR_COLUMNS},
                "has_clash": unified.get("has_clash"),
            }

            raw_path = raw_array_path(backend, cif)
            if raw_path is not None:
                if raw_path.exists() and raw_path not in raw_files_seen:
                    bytes_before += raw_path.stat().st_size
                    raw_files_seen.add(raw_path)
                arrays, topology = load_array_payload(backend, cif, raw_path)
                for key, arr in arrays.items():
                    h5f.create_dataset(f"{array_key}/{key}", data=arr,
                                        compression="gzip", compression_opts=4, shuffle=True)
                    array_datasets += 1
                if arrays:
                    row["has_array"] = True
                if topology and backend not in topology_written:
                    for tkey, tval in topology.items():
                        h5f.create_dataset(f"_topology/{backend}/{tkey}", data=tval,
                                            compression="gzip", compression_opts=4)
                    topology_written.add(backend)

            rows.append(row)

    df = pd.DataFrame(rows)
    for col in SCALAR_COLUMNS:
        df[col] = df[col].astype("float32")
    write_parquet_with_metadata(
        df, parquet_path,
        table_description=MODEL_METADATA_TABLE_DESCRIPTION,
        column_descriptions=MODEL_METADATA_COLUMN_DESCRIPTIONS,
    )

    return {
        "run": run, "n_rows": len(df), "n_arrays": array_datasets,
        "bytes_before": bytes_before,
        "h5_bytes": h5_path.stat().st_size,
        "parquet_bytes": parquet_path.stat().st_size,
    }


# ── Verification ─────────────────────────────────────────────────────────────

def verify_run(run: str, abcfold_output_root: Path, out_root: Path, sample_checks: int) -> list[str]:
    """Sanity-check the just-written parquet+HDF5 for one run. Returns a
    list of problem descriptions (empty = clean)."""
    problems = []
    predictions_dir = abcfold_output_root / run
    out_dir = out_root / run
    parquet_path = out_dir / "model_metadata.parquet"
    h5_path = out_dir / "arrays.h5"

    cifs = discover_predictions(predictions_dir)
    df = pd.read_parquet(parquet_path)
    if len(df) != len(cifs):
        problems.append(f"row count {len(df)} != {len(cifs)} CIFs discovered under {predictions_dir}")

    has_array_rows = df[df["has_array"]]
    if len(has_array_rows) == 0:
        return problems

    n = min(sample_checks, len(has_array_rows))
    sample = has_array_rows.sample(n=n, random_state=0)

    with h5py.File(h5_path, "r") as h5f:
        for _, row in sample.iterrows():
            key = row["array_key"]
            if f"{key}/pae" not in h5f:
                problems.append(f"{key}: missing pae dataset in arrays.h5")
                continue
            h5_pae = h5f[f"{key}/pae"][()]

            cif_path = abcfold_output_root / row["cif_path"]
            backend = row["backend"]
            raw_path = raw_array_path(backend, cif_path)
            if raw_path is None or not raw_path.exists():
                problems.append(f"{key}: source raw file no longer available to verify against")
                continue

            if backend == "chai1":
                m = re.search(r"model_idx_(\d+)", cif_path.name)
                src_pae = np.load(raw_path)[int(m.group(1))].astype(np.float32)
            else:
                src_pae = np.asarray(_load_json(raw_path)["pae"], dtype=np.float32)

            if h5_pae.shape != src_pae.shape:
                problems.append(f"{key}: pae shape mismatch {h5_pae.shape} vs source {src_pae.shape}")
                continue
            if not np.allclose(h5_pae.astype(np.float32), src_pae, atol=0.05):
                problems.append(f"{key}: pae values diverge beyond float16 tolerance vs source")

    return problems


# ── Deletion (only after verification passes, only with --delete-originals) ───

def delete_originals(run: str, abcfold_output_root: Path, out_root: Path) -> int:
    """Remove the raw files results/metadata/<run>/ now replaces, plus the
    always-safe cache/junk items documented in the module docstring. Only
    called by main() after verify_run() returned zero problems for this run.
    Returns bytes freed."""
    predictions_dir = abcfold_output_root / run
    parquet_path = out_root / run / "model_metadata.parquet"
    df = pd.read_parquet(parquet_path)

    freed = 0

    def rm_file(p: Path):
        nonlocal freed
        if p.exists() and p.is_file():
            freed += p.stat().st_size
            p.unlink()

    def rm_tree(p: Path):
        nonlocal freed
        if p.exists() and p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    freed += f.stat().st_size
            import shutil
            shutil.rmtree(p)

    # 1) big per-sample confidence files this script converted
    seen_raw: set[Path] = set()
    for _, row in df[df["has_array"]].iterrows():
        cif_path = abcfold_output_root / row["cif_path"]
        raw_path = raw_array_path(row["backend"], cif_path)
        if raw_path is not None and raw_path not in seen_raw:
            rm_file(raw_path)
            seen_raw.add(raw_path)

    # 2) Boltz backend-native extras outside the unified schema
    for p in predictions_dir.rglob("pde_*.npz"):
        rm_file(p)
    for p in predictions_dir.rglob("plddt_*.npz"):
        rm_file(p)

    # 3) AF3 per-seed distogram (zero consumers)
    for p in predictions_dir.glob("alphafold3_*/seed-*_distogram"):
        rm_tree(p)

    # 4) Boltz input-featurization cache + training log noise
    for p in predictions_dir.glob("boltz_*/boltz_results_*/processed"):
        rm_tree(p)
    for p in predictions_dir.glob("boltz_*/boltz_results_*/lightning_logs"):
        rm_tree(p)

    # 5) Chai-1 input templates + redundant per-model PAE copies
    for p in predictions_dir.glob("chai1_*/chai_output_*/templates"):
        rm_tree(p)
    for p in predictions_dir.glob("chai1_*/chai_output_*/pae_scores_model_*.npy"):
        rm_file(p)

    # 6) macOS junk + AF3's redundant top-level input-feature restatement
    for p in predictions_dir.rglob(".DS_Store"):
        rm_file(p)
    for p in predictions_dir.glob("alphafold3_*/*_data.json"):
        rm_file(p)

    return freed


# ── Merge ─────────────────────────────────────────────────────────────────────

ALL_MODELS_TABLE_DESCRIPTION = (
    "Concatenation of every run's model_metadata.parquet -- one row per sample "
    "across all proteins x forms x backends x seeds. See "
    "MODEL_METADATA_TABLE_DESCRIPTION / each run's own model_metadata.parquet "
    "for column meaning; see results/metadata/<protein>__<form>/arrays.h5 for "
    "the full pae/contact_probs/atom_plddt arrays a row's array_key points to."
)


def merge_all(out_root: Path) -> Path | None:
    parts = sorted(out_root.glob("*/model_metadata.parquet"))
    if not parts:
        return None
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    out_path = out_root / "all_models.parquet"
    write_parquet_with_metadata(
        df, out_path,
        table_description=ALL_MODELS_TABLE_DESCRIPTION,
        column_descriptions=MODEL_METADATA_COLUMN_DESCRIPTIONS,
    )
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    abcfold_output_root = Path(args.abcfold_output_root)
    out_root = Path(args.out_root)

    if args.protein:
        runs = args.protein
    else:
        runs = sorted(
            p.name for p in abcfold_output_root.iterdir()
            if p.is_dir() and (p / "prediction.done").exists()
        )
    if not runs:
        raise RuntimeError(f"No runs to convert under {abcfold_output_root}")

    total_before = total_h5 = total_parquet = total_freed = 0
    failures = []

    for run in runs:
        print(f"=== {run} ===")
        try:
            stats = process_run(run, abcfold_output_root, out_root)
        except Exception as e:
            print(f"  FAILED (conversion): {e}")
            failures.append(run)
            continue

        problems = verify_run(run, abcfold_output_root, out_root, args.sample_checks)
        if problems:
            print(f"  FAILED (verification): {len(problems)} problem(s)")
            for msg in problems[:10]:
                print(f"    - {msg}")
            failures.append(run)
            continue

        after = stats["h5_bytes"] + stats["parquet_bytes"]
        print(f"  {stats['n_rows']} samples, {stats['n_arrays']} arrays -> "
              f"{stats['bytes_before'] / 1e6:.1f}MB raw -> {after / 1e6:.1f}MB "
              f"(h5={stats['h5_bytes'] / 1e6:.1f}MB, parquet={stats['parquet_bytes'] / 1e6:.2f}MB), "
              f"verified clean ({args.sample_checks} spot-checks)")
        total_before += stats["bytes_before"]
        total_h5 += stats["h5_bytes"]
        total_parquet += stats["parquet_bytes"]

        if args.delete_originals:
            freed = delete_originals(run, abcfold_output_root, out_root)
            total_freed += freed
            print(f"  deleted originals: {freed / 1e6:.1f}MB freed")

    print(
        f"\n[compress] {len(runs) - len(failures)}/{len(runs)} run(s) converted. "
        f"Raw arrays: {total_before / 1e9:.2f}GB -> stored: {(total_h5 + total_parquet) / 1e9:.2f}GB"
        + (f", freed: {total_freed / 1e9:.2f}GB" if args.delete_originals else "")
    )

    if not args.skip_merge:
        merged = merge_all(out_root)
        if merged:
            print(f"[compress] merged metadata: {merged} ({merged.stat().st_size / 1e6:.1f}MB)")

    if failures:
        print(f"\n[compress] {len(failures)}/{len(runs)} run(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
