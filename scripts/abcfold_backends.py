#!/usr/bin/env python3
"""
scripts/abcfold_backends.py
=============================
Shared ABCfold output-layout knowledge for scripts/tm_helix_alignment.py
and scripts/compress_abcfold_metadata.py: which backend produced a given
CIF, how to parse its seed/sample id out of the path, and where each
backend's confidence files live relative to that CIF. Kept in one place so
the two consumers can't drift apart on backend-specific file-naming quirks
(each confirmed against a real completed run — see the docstring on each
finder below).

ABCfold writes each backend's results into its own subdirectory under
<output_dir>, named from source (per abcfold/output/*.py in the ABCFold
source): alphafold3_<name>*, boltz_results_<name>* (or boltz_*), chai_output_<name>*
(or chai1_*/chai_*), openfold_results_<name>* (or openfold3_*),
protenix_results_<name>*, rosettafold_results_<name>* (or rosettafold3_*).
"""

import re
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]

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
    across all backend subdirectories, deduplicated down to one CIF per
    (backend, seed, sample) so every backend contributes the same number of
    frames (20 seeds x 5 samples = 100). Templates fetched by
    scripts/fetch_mmseqs2_msa.py are cached elsewhere (data/fold_inputs/),
    never under results/abcfold/, so no template-CIF exclusion is needed
    here — but any stray 'templates' dir is skipped defensively anyway.

    Two backend quirks are collapsed here rather than left for
    parse_frame_id() to silently double- (or 101x-) count:
      - AlphaFold3 writes one extra '<protein>_model.cif' at the top of its
        output dir, and RosettaFold3 writes one more of the same per seed
        dir — each is a reformatted copy of that run/seed's single
        best-ranked sample (identical coordinates, cleaned-up mmCIF
        fields), not a distinct conformer. Excluded (confirmed by diffing
        one against its source sample: same Ca xyz to 3dp, only occupancy/
        entity-id formatting differs).
      - OpenFold3 and RosettaFold3 each write both a raw '..._model.cif'
        and a cleaned-up '..._model_fixed.cif' per (seed, sample) — same
        atom count and coordinates, two files on disk. Only one is kept
        per sample, preferring '_fixed' (the corrected one) when present.
    """
    all_cifs = sorted(
        c for c in predictions_dir.rglob("*.cif") if "templates" not in c.parts
    )

    best_of_run_or_seed = f"{predictions_dir.name}_model"
    per_sample = [c for c in all_cifs if c.stem != best_of_run_or_seed]

    deduped: dict[tuple[Path, str], Path] = {}
    for c in per_sample:
        key = (c.parent, strip_model_suffix(c.stem))
        if key not in deduped or c.stem.endswith("_fixed"):
            deduped[key] = c
    return sorted(deduped.values())


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


def as_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def strip_model_suffix(stem: str) -> str:
    """'<base>_model' / '<base>_model_fixed' -> '<base>' (AlphaFold3,
    RosettaFold3 and OpenFold3 all write a same-named confidence file next
    to the cif with the '_model'/'_model_fixed' suffix removed)."""
    return re.sub(r"_model(_fixed)?$", "", stem)


# ── Small scalar summary/confidence files (ptm/iptm + friends) ─────────────
# One JSON per sample, a few hundred bytes to a few KB — cheap, always kept.

def summary_path_af3_style(cif_path: Path) -> Path:
    """AlphaFold3 and RosettaFold3 write one '<base>_summary_confidences.json'
    per sample, shared by both the '_model.cif' and '_model_fixed.cif'
    variants of that sample (confirmed against a real run of each:
    results/abcfold/<protein>/{alphafold3,rosettafold}_<protein>/.../
    <base>_summary_confidences.json, top-level keys 'ptm'/'iptm')."""
    base = strip_model_suffix(cif_path.stem)
    return cif_path.with_name(f"{base}_summary_confidences.json")


def summary_path_boltz(cif_path: Path) -> Path:
    """Boltz writes 'confidence_<cif_stem>.json' (prefixed, not suffixed --
    confirmed: .../predictions/<name>/confidence_<name>_model_N.json next to
    <name>_model_N.cif), top-level keys 'ptm'/'iptm'."""
    return cif_path.with_name(f"confidence_{cif_path.stem}.json")


def summary_path_chai1(cif_path: Path) -> Path:
    """Chai-1 stores confidence in a NumPy .npz sibling, not JSON: 'pred.
    model_idx_N.cif' pairs with 'scores.model_idx_N.npz', with flat 'ptm'/
    'iptm' float arrays (confirmed against a real run)."""
    return cif_path.with_name(cif_path.name.replace("pred.", "scores.", 1)).with_suffix(".npz")


def summary_path_openfold3(cif_path: Path) -> Path:
    """OpenFold3 writes '<base>_confidences_aggregated.json' (base = the
    sample name with the '_model'/'_model_fixed' cif suffix stripped, e.g.
    '..._sample_1_confidences_aggregated.json' next to
    '..._sample_1_model.cif'), top-level keys 'ptm'/'iptm'. The plain
    '<base>_confidences.json' sibling (also written) only has raw per-atom
    pae/plddt arrays, no scalar ptm/iptm."""
    base = strip_model_suffix(cif_path.stem)
    return cif_path.with_name(f"{base}_confidences_aggregated.json")


def summary_path_protenix(cif_path: Path) -> Path | None:
    """Protenix writes '<base>_summary_confidence_sample_N.json' next to
    '<base>_sample_N.cif' (summary_confidence_ inserted before sample_N,
    not appended), top-level keys 'ptm'/'iptm'."""
    m = re.match(r"(.+)_sample_(\d+)$", cif_path.stem)
    if not m:
        return None
    base, n = m.groups()
    return cif_path.with_name(f"{base}_summary_confidence_sample_{n}.json")


def _confidence_af3_style(cif_path: Path) -> tuple[float, float]:
    candidate = summary_path_af3_style(cif_path)
    if not candidate.exists():
        return float("nan"), float("nan")
    import json
    data = json.loads(candidate.read_text())
    return as_float(data.get("ptm")), as_float(data.get("iptm"))


def _confidence_boltz(cif_path: Path) -> tuple[float, float]:
    candidate = summary_path_boltz(cif_path)
    if not candidate.exists():
        return float("nan"), float("nan")
    import json
    data = json.loads(candidate.read_text())
    return as_float(data.get("ptm")), as_float(data.get("iptm"))


def _confidence_chai1(cif_path: Path) -> tuple[float, float]:
    candidate = summary_path_chai1(cif_path)
    if not candidate.exists():
        return float("nan"), float("nan")
    with np.load(candidate) as data:
        ptm = as_float(data["ptm"][0]) if "ptm" in data.files else float("nan")
        iptm = as_float(data["iptm"][0]) if "iptm" in data.files else float("nan")
    return ptm, iptm


def _confidence_openfold3(cif_path: Path) -> tuple[float, float]:
    candidate = summary_path_openfold3(cif_path)
    if not candidate.exists():
        return float("nan"), float("nan")
    import json
    data = json.loads(candidate.read_text())
    return as_float(data.get("ptm")), as_float(data.get("iptm"))


def _confidence_protenix(cif_path: Path) -> tuple[float, float]:
    candidate = summary_path_protenix(cif_path)
    if candidate is None or not candidate.exists():
        return float("nan"), float("nan")
    import json
    data = json.loads(candidate.read_text())
    return as_float(data.get("ptm")), as_float(data.get("iptm"))


CONFIDENCE_FINDERS = {
    "alphafold3":   _confidence_af3_style,
    "rosettafold3": _confidence_af3_style,
    "boltz":        _confidence_boltz,
    "chai1":        _confidence_chai1,
    "openfold3":    _confidence_openfold3,
    "protenix":     _confidence_protenix,
}

SUMMARY_PATH_FINDERS = {
    "alphafold3":   summary_path_af3_style,
    "rosettafold3": summary_path_af3_style,
    "boltz":        summary_path_boltz,
    "chai1":        summary_path_chai1,
    "openfold3":    summary_path_openfold3,
    "protenix":     summary_path_protenix,
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
            print(f"[abcfold_backends] WARNING: no confidence finder for backend {model!r} - ptm/iptm will be NaN")
            _WARNED_BACKENDS.add(model)
        return float("nan"), float("nan")
    try:
        return finder(cif_path)
    except Exception as e:
        if model not in _WARNED_BACKENDS:
            print(f"[abcfold_backends] WARNING: confidence lookup failed for backend {model!r} ({e}) - ptm/iptm will be NaN")
            _WARNED_BACKENDS.add(model)
        return float("nan"), float("nan")


# ── Large per-sample raw confidence files (full pae/contact_probs/plddt) ──
# 12-17MB of pretty-printed JSON per sample for AF3/RosettaFold3/OpenFold3/
# Boltz (Boltz's is JSON with a stale '.npz' extension — ABCfold's own
# XOutput.pae_to_af3() overwrites the original raw PAE file in place, see
# scripts/generate_abcfold_visuals.py's module docstring point 2); Protenix
# names it 'full_data' instead of 'confidences'; Chai-1 keeps all 5 models'
# PAE matrices in one real (non-JSON) float32 .npy per seed, no
# atom_plddts/contact_probs sibling (Chai's per-atom pLDDT lives in the
# CIF's own B-factor column instead).
#
# All six keys below match AF3's own confidences.json schema exactly for
# every backend except Chai-1 — confirmed by direct inspection of AF3,
# Boltz, OpenFold3 and Protenix's raw confidence files against a real run.

def raw_confidence_path_af3_style(cif_path: Path) -> Path:
    base = strip_model_suffix(cif_path.stem)
    return cif_path.with_name(f"{base}_confidences.json")


def raw_confidence_path_boltz(cif_path: Path) -> Path:
    return cif_path.with_name(f"pae_{cif_path.stem}.npz")


def raw_confidence_path_openfold3(cif_path: Path) -> Path:
    base = strip_model_suffix(cif_path.stem)
    return cif_path.with_name(f"{base}_confidences.json")


def raw_confidence_path_protenix(cif_path: Path) -> Path | None:
    m = re.match(r"(.+)_sample_(\d+)$", cif_path.stem)
    if not m:
        return None
    base, n = m.groups()
    return cif_path.with_name(f"{base}_full_data_sample_{n}.json")


def raw_confidence_path_chai1(cif_path: Path) -> Path:
    """Chai-1 keeps every model's PAE for a given seed in one array at
    <seed_dir>/pae_scores.npy, shape (n_models, N, N) float32 — index by
    the model index parsed out of 'pred.model_idx_N.cif'."""
    return cif_path.with_name("pae_scores.npy")


RAW_CONFIDENCE_PATH_FINDERS = {
    "alphafold3":   raw_confidence_path_af3_style,
    "rosettafold3": raw_confidence_path_af3_style,
    "boltz":        raw_confidence_path_boltz,
    "openfold3":    raw_confidence_path_openfold3,
    "protenix":     raw_confidence_path_protenix,
    "chai1":        raw_confidence_path_chai1,
}
