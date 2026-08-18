#!/usr/bin/env python3
"""
rescoring/src/fit_pocket_lda.py
==================================
Per-position, per-Z-scale shrinkage-LDA coefficients for each ligand
category this pipeline co-folds — the sequence-only half of the "Rosetta
hotspots vs. sequence-LDA importance" overlay `aggregate.py`/`plots.py`
produce (see worflows/postprocessing/Snakefile's stage 12 comment for the
full rationale on why this is a single-group overlay, not an
importer-vs-non-importer Rosetta comparison).

Same Z-scale encoding + `LinearDiscriminantAnalysis(solver="lsqr",
shrinkage="auto")` fit as `NPF_LDA_kernel/workflow/scripts/analyse_track_a.py`
(copied here, not imported — NPF_LDA_kernel is a sibling project, not a
dependency of this one), applied to `NPF_LDA_kernel`'s own
`pocket_sites_cdd_msa.tsv` pocket strings (same 35-position pocket every
`build_position_mapping.py` row is keyed on), against two different label
sources depending on the ligand:

- **GA1**: labels come from `NPF_LDA_kernel`'s own real, assay-based
  `results/ga_classifier/labels.tsv` (45 proteins, Chiba et al. 2015 +
  Jorgensen et al. 2017) — the one ligand this pipeline has a genuine
  published importer/non-importer classifier for. This script still does
  the fitting itself (NPF_LDA_kernel only ever ran this exact Z-scale/LDA
  fit for its own "hc" 33-protein subset, not the full 45), but the LABELS
  are the real published ones, not a heuristic.
- **Every other ligand with enough assigned proteins**: one-vs-rest labels
  built from `scripts/ligand_assignment.py`'s `LIGAND_GROUPS` (this
  pipeline's own protein->ligand assignment) — "is this protein's pocket
  more like the proteins actually co-folded with ligand X, or not" — fit
  against the full 53-protein corpus (every non-X protein, including
  apoform-only ones, is a valid negative). Requires MIN_POSITIVES (5)
  assigned proteins to attempt a fit at all; below that (auxin, glycerate,
  dimethylarsenate, JA-Ile, glycylglycine, spermidine,
  quercetin-3-O-sophoroside — 1-2 proteins each in this corpus) a
  meaningful LDA can't be fit, so no output is written for that ligand —
  `aggregate.py`'s overlay is Rosetta-hotspot-only for those.

Statistical validation (LOO-AUC, permutation p-value) is intentionally out
of scope here — that's what NPF_LDA_kernel's own workflow already does for
the GA1 classifier; these one-vs-rest refits exist for overlay/
interpretation against this pipeline's own Rosetta energetics, not to be
published as classifiers in their own right.

Usage:
    python fit_pocket_lda.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import config

sys.path.insert(0, str(config.PIPELINE_ROOT / "scripts"))
from ligand_assignment import LIGAND_GROUPS  # noqa: E402

MIN_POSITIVES = 5

# Z-scales (Sandberg et al. 1998, Hellberg 1987) -- identical to
# NPF_LDA_kernel/workflow/scripts/npf_ga_classifier.py / analyse_track_a.py.
ZSCALES = {
    "A": ( 0.24, -2.32,  0.60, -0.14,  1.30),
    "R": ( 3.52,  2.50, -3.50,  1.99, -0.17),
    "N": ( 3.05,  1.62,  1.04, -1.15,  1.61),
    "D": ( 3.98,  0.93,  1.93, -2.46,  0.75),
    "C": ( 0.84, -1.67,  3.71,  0.18, -2.65),
    "Q": ( 1.75,  0.50, -1.44, -1.34,  0.66),
    "E": ( 3.11,  0.26, -0.11, -3.04, -0.25),
    "G": ( 2.05, -4.06,  0.36, -0.82, -0.38),
    "H": ( 2.47,  1.95,  0.26,  3.90,  0.09),
    "I": (-3.89, -1.73, -1.71, -0.84,  0.26),
    "L": (-4.28, -1.30, -1.49, -0.72,  0.84),
    "K": ( 2.29,  0.89, -2.49,  1.49,  0.31),
    "M": (-2.85, -0.22,  0.47,  1.94, -0.98),
    "F": (-4.22,  1.94,  1.06,  0.54, -0.62),
    "P": (-1.66,  0.27,  1.84,  0.70,  2.00),
    "S": ( 2.39, -1.07,  1.15, -1.39,  0.67),
    "T": ( 0.75, -2.18, -1.12, -1.46, -0.40),
    "V": (-2.59, -2.64, -1.54, -0.85, -0.02),
    "W": (-4.36,  3.94,  0.59,  3.44, -1.59),
    "Y": (-2.54,  2.44,  0.43,  0.04, -1.47),
    "X": ( 0.00,  0.00,  0.00,  0.00,  0.00),  # gap
}
ZDIM = 5
Z_NAMES = ["Z1_hydrophil", "Z2_steric", "Z3_electronic", "Z4_electro", "Z5_proline"]


def encode(pocket: str) -> np.ndarray:
    out = []
    for ch in pocket.upper():
        out.extend(ZSCALES.get(ch, (0.0,) * ZDIM))
    return np.asarray(out, dtype=float)


def load_pockets() -> dict[str, str]:
    if not config.NPF_LDA_KERNEL_POCKET_SITES.exists():
        sys.exit(f"{config.NPF_LDA_KERNEL_POCKET_SITES} not found.")
    pockets = {}
    for line in config.NPF_LDA_KERNEL_POCKET_SITES.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, pocket = line.split("\t")
        pockets[name] = pocket
    return pockets


def load_ga1_labels() -> dict[str, int]:
    if not config.NPF_LDA_KERNEL_GA_LABELS.exists():
        sys.exit(f"{config.NPF_LDA_KERNEL_GA_LABELS} not found.")
    labels = {}
    for line in config.NPF_LDA_KERNEL_GA_LABELS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, val = line.split()
        labels[name] = int(val)
    return labels


def fit_and_write(ligand_key: str, names: list[str], y: np.ndarray, pockets: dict[str, str]) -> None:
    X = np.vstack([encode(pockets[n]) for n in names])
    L = len(pockets[names[0]])

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    coef5 = clf.coef_.ravel().reshape(L, ZDIM)

    loading_rows = [
        {"position": pos + 1, "z_dim": z + 1, "z_name": Z_NAMES[z], "lda_coef": round(float(coef5[pos, z]), 4)}
        for pos in range(L) for z in range(ZDIM)
    ]
    loadings_path = config.DATA_DIR / f"lda_{ligand_key}_loadings.tsv"
    pd.DataFrame(loading_rows).to_csv(loadings_path, sep="\t", index=False)

    importance = np.abs(coef5).sum(axis=1)
    importance_rows = [{"position": pos + 1, "importance": round(float(importance[pos]), 6)} for pos in range(L)]
    importance_path = config.DATA_DIR / f"position_importance_{ligand_key}.tsv"
    pd.DataFrame(importance_rows).to_csv(importance_path, sep="\t", index=False)

    print(f"[fit_pocket_lda] {ligand_key}: n={len(y)} (pos={int(y.sum())}, neg={len(y) - int(y.sum())}) "
          f"-> {loadings_path.name}, {importance_path.name}")


def main():
    pockets = load_pockets()
    corpus = sorted(pockets)
    npf_to_full = {name.rsplit("_", 1)[0]: name for name in corpus}

    # -- GA1: real, assay-based labels, fit against every labeled protein --
    ga1_labels = load_ga1_labels()
    ga1_names = [n for n in ga1_labels if n in pockets]
    missing = set(ga1_labels) - set(ga1_names)
    if missing:
        print(f"[fit_pocket_lda] GA1: {len(missing)} labeled protein(s) not in this "
              f"corpus's pocket table, skipped: {sorted(missing)}")
    y = np.array([ga1_labels[n] for n in ga1_names])
    fit_and_write("GA1", ga1_names, y, pockets)

    # -- everything else: one-vs-rest from this pipeline's own ligand assignment --
    for ligand_key, npf_names in LIGAND_GROUPS.items():
        if ligand_key == "GA1":
            continue  # already handled above with the real classifier's labels
        positives = sorted({npf_to_full[n] for n in npf_names if n in npf_to_full})
        if len(positives) < MIN_POSITIVES:
            print(f"[fit_pocket_lda] {ligand_key}: only {len(positives)} assigned protein(s) "
                  f"in this corpus (< MIN_POSITIVES={MIN_POSITIVES}) -- no LDA fit, "
                  "Rosetta-hotspot-only in the overlay")
            continue
        names = corpus
        y = np.array([1 if n in set(positives) else 0 for n in names])
        fit_and_write(ligand_key, names, y, pockets)

    print("[fit_pocket_lda] done.")


if __name__ == "__main__":
    main()
