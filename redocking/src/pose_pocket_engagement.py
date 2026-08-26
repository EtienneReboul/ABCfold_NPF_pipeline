#!/usr/bin/env python3
"""
redocking/src/pose_pocket_engagement.py
===========================================
Stage 8: is HADDOCK3's own top-HADDOCK-score model actually a reliable pick,
or can a genuinely off-target ("membrane"/non-pocket) pose outscore every
real pocket-engaging pose in a given complex?

`compare_to_abcfold.py`'s non-importer check only ever looks at the top-4
models by HADDOCK score. Found by hand (2026-08-26, prompted by the user
eyeballing `haddock_redocking_exploration.ipynb`'s non-importer pocket-
contact box plot): pooling those top-4-per-complex contact counts across
all 56 non-importer complexes shows a sharp, unmistakable bimodal split --
46/228 (20%) have LITERALLY ZERO CDD active-residue contacts, a huge spike
disconnected from the rest of the distribution (which is a broad hump
around 10-15 contacts), and contact count only correlates -0.43 with
HADDOCK score -- i.e. a real fraction of genuinely off-target poses can
still out-score real pocket-engaging ones. This script scores EVERY kept
`[flexref]` model (not just the top-4) for EVERY complex (both roles, not
just non-importer -- the same contamination risk applies to importers'
"top-ranked model" selection that every other downstream script trusts),
then fits a 2-component 1-D Gaussian mixture on pooled contact counts to
classify each pose as pocket-engaging ("good") or off-target ("bad") --
exactly the kind of unsupervised split the raw histogram already shows by
eye, just made reproducible/quantitative instead of an eyeballed cutoff.

**Why GMM(2) here rather than LDA**: there is no labeled ground truth for
"this specific pose is/isn't in the real pocket" to train a discriminant
on -- this is an unsupervised mode-separation problem on one feature
(contact count), which is exactly what a Gaussian mixture is for. LDA
needs known class labels per sample; the only labels available here
(importer/non_importer) are per-*complex*, not per-*pose*, and are
deliberately NOT used to fit this split (the whole point is a role-
agnostic "is this pose physically in the pocket" filter, not a role
classifier -- mixing the two would just reintroduce the confound the CDD-
restraint-driven docking protocol already has, see
project_redocking_pipeline_plan memory).

Output:
  results/comparison/pose_pocket_engagement.csv   every kept model, every
    complex: complex_id, protein, role, ca_cluster, caprieval_rank,
    haddock_score, n_active_residues_contacted, n_active_residues_total,
    good_pose (bool, from the fitted GMM)
  results/comparison/good_pose_representative.csv   per complex: the best-
    HADDOCK-score model AMONG that complex's good_pose==True models (falls
    back to the plain global best if a complex has zero good poses, with
    n_good_poses=0 flagged so callers can tell the difference) -- the
    corrected replacement for compare_to_abcfold.py's/
    rescore_redocked_batch.py's plain top_ranked_model() selection.

Run after redocking's Stage 6 (HADDOCK3 array) has completed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import config
from compare_to_abcfold import (
    LIGAND_CHAIN_HADDOCK,
    POCKET_CONTACT_CUTOFF,
    _find_final_caprieval_dir,
    _ligand_heavy_coords,
    _model_path,
    _read_capri_ss,
)


def full_ensemble_contacts(complex_id: str, protein: str, run_dir: Path) -> list[dict]:
    """Every kept model's CDD active-residue contact count for one complex
    -- same contact definition as compare_to_abcfold.compare_non_importer,
    just applied to every row of capri_ss.tsv instead of only the top-N."""
    active_residues = set(config.load_cdd_residues(protein))
    caprieval_dir = _find_final_caprieval_dir(run_dir)
    rows = _read_capri_ss(caprieval_dir)

    results = []
    for row in rows:
        model_path = _model_path(caprieval_dir, row)
        st = gemmi.read_structure(str(model_path))
        st.setup_entities()
        ligand_coords = _ligand_heavy_coords(st, LIGAND_CHAIN_HADDOCK)

        contacted = set()
        for model in st:
            for chain in model:
                if chain.name != config.PROTEIN_CHAIN:
                    continue
                for res in chain:
                    if res.seqid.num not in active_residues:
                        continue
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                        if np.any(np.linalg.norm(ligand_coords - p, axis=1) <= POCKET_CONTACT_CUTOFF):
                            contacted.add(res.seqid.num)
            break
        results.append({
            "complex_id": complex_id, "protein": protein,
            "caprieval_rank": int(row["caprieval_rank"]), "haddock_score": float(row["score"]),
            "n_active_residues_contacted": len(contacted), "n_active_residues_total": len(active_residues),
        })
    return results


def fit_good_pose_gmm(contacts: np.ndarray, seed: int = 42) -> GaussianMixture:
    gmm = GaussianMixture(n_components=2, n_init=10, random_state=seed)
    gmm.fit(contacts.reshape(-1, 1))
    return gmm


def classify_good_pose(df: pd.DataFrame, gmm: GaussianMixture) -> pd.Series:
    """True for the higher-mean ("pocket-engaging") GMM component."""
    good_component = int(np.argmax(gmm.means_.flatten()))
    labels = gmm.predict(df["n_active_residues_contacted"].values.reshape(-1, 1))
    return labels == good_component


def main() -> None:
    with config.MANIFEST_CSV.open() as f:
        manifest_rows = list(csv.DictReader(f))

    frames = []
    for row in manifest_rows:
        complex_id, protein, role = row["complex_id"], row["protein"], row["role"]
        run_dir = config.HADDOCK_RUNS_DIR / complex_id
        try:
            ensemble = full_ensemble_contacts(complex_id, protein, run_dir)
        except Exception as exc:
            print(f"{complex_id}: FAILED -- {exc}")
            continue
        for r in ensemble:
            r["role"] = role
            r["ca_cluster"] = row["ca_cluster"]
        frames.append(pd.DataFrame(ensemble))
        print(f"{complex_id}: {len(ensemble)} models scored")

    all_poses = pd.concat(frames, ignore_index=True)

    gmm = fit_good_pose_gmm(all_poses["n_active_residues_contacted"].values)
    all_poses["good_pose"] = classify_good_pose(all_poses, gmm)

    means = sorted(gmm.means_.flatten())
    weights = gmm.weights_[np.argsort(gmm.means_.flatten())]
    print(f"\nGMM(2) on pooled n_active_residues_contacted (n={len(all_poses)}): "
          f"component means={means[0]:.2f}/{means[1]:.2f}, weights={weights[0]:.2f}/{weights[1]:.2f}")

    out_csv = config.COMPARISON_DIR / "pose_pocket_engagement.csv"
    all_poses.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(all_poses)} rows)")

    for role in ["importer", "non_importer"]:
        sub = all_poses[all_poses["role"] == role]
        frac_bad = 1 - sub["good_pose"].mean()
        n_complexes = sub["complex_id"].nunique()
        zero_good = sub.groupby("complex_id")["good_pose"].any()
        n_zero_good = int((~zero_good).sum())
        print(f"{role}: {frac_bad:.1%} of poses classified 'bad' (off-target); "
              f"{n_zero_good}/{n_complexes} complexes have ZERO good poses among all kept models")

    rep_rows = []
    for complex_id, group in all_poses.groupby("complex_id"):
        good = group[group["good_pose"]]
        n_good = len(good)
        pool = good if n_good else group
        best = pool.loc[pool["haddock_score"].idxmin()]
        rep_rows.append({
            "complex_id": complex_id, "protein": best["protein"], "role": best["role"],
            "caprieval_rank": int(best["caprieval_rank"]), "haddock_score": best["haddock_score"],
            "n_active_residues_contacted": int(best["n_active_residues_contacted"]),
            "n_good_poses": n_good,
        })
    rep_df = pd.DataFrame(rep_rows).sort_values("complex_id")
    rep_csv = config.COMPARISON_DIR / "good_pose_representative.csv"
    rep_df.to_csv(rep_csv, index=False)
    n_changed = (rep_df["caprieval_rank"] != 1).sum()
    print(f"Wrote {rep_csv} -- {n_changed}/{len(rep_df)} complexes' representative model "
          f"changed from the plain top-HADDOCK-score pick (caprieval_rank=1) after filtering")


if __name__ == "__main__":
    main()
