#!/usr/bin/env python3
"""
scripts/cluster_conformations.py
====================================
Headless, non-interactive port of scripts/_notebook_setup_functions.py's
plot_pca()/plot_ligand_pca() -- runs the same two-stage clustering (macro
TM-Ca conformational state, then per-macro-state ligand pose) as a
Snakemake pipeline stage instead of an interactive notebook cell, writing
static SVGs instead of Plotly FigureWidgets.

Fixed, non-interactive scheme (mirrors generate_notebook_protein_cells.py's
per-protein cell template and the already-computed on-disk
results/tm_reannotated/results/ligand_pose from manual notebook runs, so
nothing here re-derives those conventions differently):

  1. Macro-state: GMM, manual k=3, tag "pca_k3" -- run for every protein
     with a completed holoform run (apoform-only proteins have no ligand to
     sub-cluster, so this step is skipped for them: it exists to feed step 2).
     Mirrors plot_pca(protein, n_components=3).
  2. Ligand-pose: for every macro-state cluster with >= min_holo_frames (8)
     holoform members, GMM n_components="auto" (BIC-knee, auto_k_max=6) on
     that cluster's re-aligned ligand-atom PCA. Mirrors
     plot_ligand_pca(protein, "pca_k3", n_components="auto", auto_k_max=6).

Same >95%-variance 1-D histogram fallback as the notebook functions
(_plot_pca_1d_fallback / _plot_ligand_pca_1d_fallback) -- when it fires for
the macro-state step, the resulting reannotation tag is NOT "pca_k3" (it's
"pca_k{k}_hist1d", same as the notebook code), so step 2 is skipped for that
protein with a clear log message rather than failing to find assignments.parquet
under a tag that was never written -- this is a latent case in the notebook
code too (plot_ligand_pca(protein, "pca_k3", ...) would raise
FileNotFoundError there), just handled here as a skip instead of a crash.

Usage:
    python scripts/cluster_conformations.py --protein NPF2.12_Q9LFX9
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from parquet_utils import write_parquet_with_metadata  # noqa: E402

ABCFOLD_OUT_ROOT = ROOT / "results" / "abcfold"
ALIGN_ROOT = ROOT / "results" / "tm_alignment"
REANN_ROOT = ROOT / "results" / "tm_reannotated"
FIG_ROOT = ROOT / "results" / "figures"
LIGPOSE_ROOT = ROOT / "results" / "ligand_pose"

MACRO_METHOD_TAG = "pca_k3"
MACRO_N_COMPONENTS = 3
MIN_HOLO_FRAMES = 8
LIGAND_AUTO_K_MAX = 6
IPTM_THRESHOLD = 0.5

GMM_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

# Mirrors scripts/tm_helix_alignment.py's BACKEND_PATTERNS / backend_of() /
# discover_predictions() / parse_frame_id() -- duplicated here (not
# imported), same convention _notebook_setup_functions.py already uses.
BACKEND_PATTERNS = {
    "alphafold3":   "alphafold3",
    "boltz":        "boltz",
    "chai1":        "chai",
    "openfold3":    "openfold",
    "protenix":     "protenix",
    "rosettafold3": "rosettafold",
}


# ── Ensemble loading (mirrors _notebook_setup_functions.py's _load_run / _kabsch_fit / _load_protein) ──

def _load_run(run_name: str):
    npy = ALIGN_ROOT / run_name / "aligned_ca_tm.npy"
    meta_path = ALIGN_ROOT / run_name / "meta.parquet"
    if not npy.exists():
        raise FileNotFoundError(
            f"{npy} not found -- run worflows/postprocessing/Snakefile "
            f"(scripts/tm_helix_alignment.py) for {run_name} first")
    coords = np.load(npy)
    meta = pd.read_parquet(meta_path)
    X = coords.reshape(coords.shape[0], -1)
    return X, meta


def _kabsch_fit(P, Q):
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_mean - R @ p_mean
    return R, t


def _load_protein(protein: str, iptm_threshold: float = IPTM_THRESHOLD):
    """See _notebook_setup_functions.py's _load_protein docstring for the
    full rationale (ipTM filtering, holo-onto-apo Kabsch refit, unique_frame_id).
    Kept behaviorally identical here."""
    X_parts, meta_parts = [], []
    apo_mean = None
    for status in ("apo", "holo"):
        run_name = f"{protein}__{status}"
        if not (ALIGN_ROOT / run_name).exists():
            if status == "apo":
                raise FileNotFoundError(
                    f"{ALIGN_ROOT / run_name} not found -- apoform is expected "
                    f"for every protein; run worflows/postprocessing/Snakefile first")
            continue
        X, meta = _load_run(run_name)
        coords = X.reshape(X.shape[0], -1, 3)

        if status == "holo" and iptm_threshold:
            keep = (meta["iptm"] >= iptm_threshold).to_numpy()
            n_dropped = int((~keep).sum())
            if n_dropped:
                print(f"[load_protein] {run_name}: dropping {n_dropped}/{len(meta)} "
                      f"frame(s) with ipTM < {iptm_threshold}")
            if not keep.any():
                print(f"[load_protein] {run_name}: 0/{len(meta)} frame(s) pass "
                      f"ipTM >= {iptm_threshold}, skipping this holoform run entirely")
                continue
            meta, X, coords = meta.loc[keep].reset_index(drop=True), X[keep], coords[keep]

        if status == "apo":
            apo_mean = coords.mean(axis=0)
        else:
            R, t = _kabsch_fit(coords.mean(axis=0), apo_mean)
            flat = coords.reshape(-1, 3)
            coords = ((R @ flat.T).T + t).reshape(coords.shape)
            X = coords.reshape(coords.shape[0], -1)

        meta = meta.copy()
        meta["status"] = status
        meta["source_run"] = run_name
        meta["unique_frame_id"] = status + "_" + meta["frame_id"].astype(str)
        X_parts.append(X)
        meta_parts.append(meta)

    X = np.concatenate(X_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, meta


# ── CIF discovery / reannotation (mirrors _notebook_setup_functions.py) ──

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
    predictions_dir = ABCFOLD_OUT_ROOT / run_name
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


def _build_cif_by_key(meta: pd.DataFrame) -> dict[tuple[str, str], Path]:
    cif_by_key = {}
    for source_run in sorted(meta["source_run"].unique()):
        predictions_dir = ABCFOLD_OUT_ROOT / source_run
        for c in _discover_abcfold_cifs(source_run):
            cif_by_key[(source_run, _frame_id_for_cif(c, predictions_dir))] = c
    return cif_by_key


ASSIGNMENTS_TABLE_DESCRIPTION = (
    "Cluster membership for one protein's conformational ensemble (GMM), "
    "written by scripts/cluster_conformations.py's non-interactive port of "
    "_notebook_setup_functions.py's _reannotate(). One row per pooled "
    "apo+holo frame; symlinked marks whether that frame's CIF was actually "
    "symlinked into this directory's cluster_<k>/ (frames beyond "
    "max_per_cluster per cluster are still listed here, just not symlinked to disk)."
)
ASSIGNMENTS_FIXED_COLUMN_DESCRIPTIONS = {
    "protein": "Base protein identifier",
    "status": "'apo' or 'holo' -- which run this frame came from",
    "model": "Folding backend that produced this frame",
    "seed": "Random seed used for this backend run",
    "sample_index": "Sample/diffusion index within that seed",
    "frame_id": "unique_frame_id (status-prefixed) -- matches the symlinked CIF's filename stem",
    "ptm": "Predicted TM-score (pTM) for this frame",
    "iptm": "Predicted interface TM-score (ipTM)",
    "cluster": "Cluster id this frame was assigned to",
    "symlinked": "True if this frame's CIF was actually symlinked into cluster_<k>/ "
                 "(capped at max_per_cluster per cluster -- every frame is still listed "
                 "here regardless)",
}


def _assignments_column_descriptions(x_col: str, y_col: str) -> dict[str, str]:
    coord_desc = ("2D embedding coordinate this clustering was run on -- a PCA "
                  "component for the 2-D path, or the 1-D PCA value duplicated "
                  "onto both axes for the 1-D fallback path")
    return {
        **ASSIGNMENTS_FIXED_COLUMN_DESCRIPTIONS,
        x_col: coord_desc + " (x axis)",
        y_col: coord_desc + " (y axis)",
    }


LIGAND_POSE_TABLE_DESCRIPTION = (
    "Ligand-pose sub-cluster membership within one Ca-conformation cluster's "
    "holoform frames (GMM), written by scripts/cluster_conformations.py -- "
    "on an all-heavy-atom ligand PCA, fit after a fresh per-cluster TM-Ca "
    "Kabsch realignment. One row per holo frame in this one Ca cluster; "
    "'cluster' here is the ligand POSE id, NOT the parent Ca-conformation "
    "cluster id (that one is instead the 'ca_cluster_<k>' path component "
    "this table's directory sits under). 'symlinked' marks whether that "
    "frame's whole-structure CIF (protein + ligand together) was actually "
    "symlinked into this directory's cluster_<pose>/."
)


def _ligand_pose_column_descriptions(x_col: str, y_col: str) -> dict[str, str]:
    coord_desc = ("PCA coordinate of this frame's ligand all-heavy-atom xyz, "
                  "after Kabsch-realigning its Ca atoms onto this one Ca "
                  "cluster's own local mean and applying that same "
                  "transform to the ligand -- PCA fit on this Ca cluster's "
                  "holo frames only, not the whole pooled ensemble")
    return {
        **ASSIGNMENTS_FIXED_COLUMN_DESCRIPTIONS,
        "cluster": "Ligand pose id (GMM sub-cluster of this Ca cluster's ligand-atom "
                   "PCA -- NOT the parent Ca-conformation cluster id)",
        x_col: coord_desc + " (x axis)",
        y_col: coord_desc + " (y axis)",
    }


def _reannotate(protein, meta, labels, x_col, y_col, method_tag,
                 max_per_cluster=20, sample_seed=42, out_dir=None,
                 table_description=None, column_descriptions=None):
    """Verbatim port of _notebook_setup_functions.py's _reannotate (no
    plotting involved there to begin with) -- see that module for the full
    docstring/rationale."""
    cif_by_key = _build_cif_by_key(meta)

    meta = meta.copy()
    meta["gmm_cluster"] = labels
    out_dir = (REANN_ROOT / protein / method_tag) if out_dir is None else out_dir
    cluster_ids = sorted(set(int(l) for l in labels))

    assign_rows = []
    n_symlinked = 0
    for cid in cluster_ids:
        dir_name = "cluster_noise" if cid == -1 else f"cluster_{cid}"
        cluster_dir = out_dir / dir_name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for stale in cluster_dir.iterdir():
            if stale.is_symlink():
                stale.unlink()

        cluster_rows = meta[meta["gmm_cluster"] == cid]
        sampled_idx = set(cluster_rows.sample(
            n=min(len(cluster_rows), max_per_cluster), random_state=sample_seed,
        ).index)

        for idx, row in cluster_rows.iterrows():
            cif = cif_by_key.get((row["source_run"], row["frame_id"]))
            symlinked = cif is not None and idx in sampled_idx
            if symlinked:
                dest = cluster_dir / f"{row['unique_frame_id']}.cif"
                if not dest.exists():
                    dest.symlink_to(cif.resolve())
                    n_symlinked += 1
            assign_rows.append({
                "protein":      protein,
                "status":       row["status"],
                "model":        row["model"],
                "seed":         row["seed"],
                "sample_index": row["sample_index"],
                "frame_id":     row["unique_frame_id"],
                "ptm":          row["ptm"],
                "iptm":         row["iptm"],
                "cluster":      cid,
                x_col:          round(float(row[x_col]), 4),
                y_col:          round(float(row[y_col]), 4),
                "symlinked":    symlinked,
            })
    write_parquet_with_metadata(
        pd.DataFrame(assign_rows), out_dir / "assignments.parquet",
        table_description=table_description if table_description is not None else ASSIGNMENTS_TABLE_DESCRIPTION,
        column_descriptions=column_descriptions if column_descriptions is not None else _assignments_column_descriptions(x_col, y_col),
    )
    print(f"[reannotate] {protein}/{method_tag}: {n_symlinked} symlinks "
          f"(max {max_per_cluster}/cluster) of {len(assign_rows)} assignments -> {out_dir}")


def _fit_gmm_bic_sweep(xy, k_min=1, k_max=20, n_init=20, random_state=42):
    """Verbatim port of _notebook_setup_functions.py's _fit_gmm_bic_sweep."""
    k_max = min(k_max, xy.shape[0] - 1)
    ks = list(range(max(1, k_min), k_max + 1))

    gmms, bic_by_k = {}, {}
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=n_init, random_state=random_state)
        try:
            gmm.fit(xy)
        except ValueError as e:
            print(f"[gmm-auto] WARNING: k={k} failed ({e}), skipping")
            continue
        gmms[k] = gmm
        bic_by_k[k] = float(gmm.bic(xy))

    if not bic_by_k:
        raise RuntimeError(
            f"GMM auto (BIC sweep) failed for every k in [{ks[0]}, {ks[-1]}] -- "
            "try a narrower auto_k_min/auto_k_max range")
    ks = sorted(bic_by_k)
    best_k = ks[int(np.argmin([bic_by_k[k] for k in ks]))]
    if len(ks) >= 3:
        degree = min(7, max(1, len(ks) - 3))
        try:
            kl = KneeLocator(ks, [bic_by_k[k] for k in ks],
                              curve="convex", direction="decreasing",
                              interp_method="polynomial", polynomial_degree=degree)
            if kl.knee is not None:
                best_k = int(kl.knee)
        except Exception as e:
            print(f"[gmm-auto] WARNING: KneeLocator failed ({e}), falling back to BIC minimum")

    return gmms[best_k], best_k, bic_by_k


# ── Static (SVG) plotting -- go.Figure, not go.FigureWidget; no click handlers/display ──

def _save_svg(fig: go.Figure, protein: str, filename: str) -> Path:
    out_dir = FIG_ROOT / protein
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.write_image(str(path), format="svg")
    return path


def _ellipse_trace(mean, cov, color, n_std=1.5, n_pts=80):
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    t = np.linspace(0, 2 * np.pi, n_pts)
    pts = n_std * (vecs * np.sqrt(np.maximum(vals, 0))) @ np.vstack(
        [np.cos(t), np.sin(t)])
    x, y = mean[0] + pts[0], mean[1] + pts[1]
    return go.Scatter(
        x=np.append(x, x[0]), y=np.append(y, y[0]),
        mode="lines",
        line=dict(color=color, width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    )


def _plot_bic_curve_svg(protein, method_title, bic_by_k, best_k, method_tag):
    ks = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=bics, mode="lines+markers",
        line=dict(color="#1565C0", width=2), marker=dict(size=6), name="BIC",
    ))
    fig.add_trace(go.Scatter(
        x=[best_k], y=[bic_by_k[best_k]], mode="markers",
        marker=dict(size=14, color="#d62728", symbol="star"), name=f"knee k={best_k}",
    ))
    fig.update_layout(
        title=f"{protein}<br>{method_title}<br>BIC sweep k={ks[0]}-{ks[-1]}, knee k={best_k}",
        xaxis_title="n_components (k)", yaxis_title="BIC",
        template="plotly_white", height=380, width=520, showlegend=False,
    )
    _save_svg(fig, protein, f"{method_tag}_bic_k{best_k}.svg")


def _plot_gmm_embedding_svg(protein, meta, xy, x_col, y_col, k, gmm, fig_tag,
                             title, axis_titles, marker_size=6, opacity=0.7):
    """Static GMM-cluster scatter + covariance ellipses -- mirrors the GMM
    branch of _notebook_setup_functions.py's _plot_embedding, minus the
    live FigureWidget/click-to-reveal machinery."""
    fig = go.Figure()
    for i in range(k):
        sub = meta[meta["gmm_cluster"] == i]
        color = GMM_PALETTE[i % len(GMM_PALETTE)]
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode="markers",
            marker=dict(size=marker_size, color=color, opacity=opacity),
            name=f"cluster {i}",
        ))
        fig.add_trace(_ellipse_trace(gmm.means_[i], gmm.covariances_[i], color))
    fig.update_layout(
        title=title, xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
        legend_title="GMM cluster", template="plotly_white",
        height=540, width=680,
    )
    _save_svg(fig, protein, f"{fig_tag}_embedding.svg")


def _plot_hist1d_svg(protein, meta, x_col, k, title, xaxis_title, fig_tag):
    fig = go.Figure()
    for i in range(k):
        sub = meta[meta["gmm_cluster"] == i]
        color = GMM_PALETTE[i % len(GMM_PALETTE)]
        fig.add_trace(go.Histogram(x=sub[x_col], name=f"cluster {i}", marker_color=color, opacity=0.7))
    fig.update_layout(
        title=title, xaxis_title=xaxis_title, yaxis_title="count",
        barmode="overlay", legend_title="GMM cluster",
        template="plotly_white", height=430, width=680,
    )
    _save_svg(fig, protein, f"{fig_tag}_hist.svg")


# ── Stage 1: macro-state (Ca-conformation) clustering ──

def cluster_macro_state(protein: str, max_per_cluster: int = 20) -> str:
    """GMM, manual k=MACRO_N_COMPONENTS, tagged MACRO_METHOD_TAG -- mirrors
    plot_pca(protein, n_components=3). Returns the method_tag actually used
    (MACRO_METHOD_TAG normally; a different "pca_k{k}_hist1d" tag if the
    >95%-variance 1-D fallback fires -- see module docstring)."""
    X, meta = _load_protein(protein)

    n_pc = min(max(2, 5), X.shape[1])
    pca = PCA(n_components=n_pc)
    coords = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    if evr[0] > 0.95:
        pc1 = coords[:, 0]
        gmm, k_used, bic_by_k = _fit_gmm_bic_sweep(pc1.reshape(-1, 1))
        method_title = f"PCA  PC1-only fallback ({evr[0]:.1%} var > 95%)"
        _plot_bic_curve_svg(protein, method_title, bic_by_k, k_used, "pca")

        labels = gmm.predict(pc1.reshape(-1, 1))
        meta = meta.copy()
        meta["pc_x"] = pc1
        meta["gmm_cluster"] = labels
        fig_tag = f"pca_k{k_used}_hist1d"
        _plot_hist1d_svg(protein, meta, "pc_x", k_used,
                          f"{protein}<br>{method_title}<br>GMM auto (BIC knee) k={k_used}",
                          f"PC1  ({evr[0]:.1%} var)", fig_tag)
        _reannotate(protein, meta, labels, x_col="pc_x", y_col="pc_x", method_tag=fig_tag,
                    max_per_cluster=max_per_cluster)
        print(f"[macro-state] {protein}: >95% PC1 variance -- used 1-D fallback "
              f"(tag={fig_tag!r}), NOT {MACRO_METHOD_TAG!r} -- ligand-pose step will be skipped")
        return fig_tag

    xy = coords[:, [0, 1]]
    gmm = GaussianMixture(n_components=MACRO_N_COMPONENTS, covariance_type="full",
                           n_init=20, random_state=42)
    gmm.fit(xy)
    labels = gmm.predict(xy)

    meta = meta.copy()
    meta["pc_x"], meta["pc_y"] = xy[:, 0], xy[:, 1]
    meta["gmm_cluster"] = labels

    _plot_gmm_embedding_svg(
        protein, meta, xy, "pc_x", "pc_y", MACRO_N_COMPONENTS, gmm, MACRO_METHOD_TAG,
        title=f"{protein}<br>PCA  PC1 vs PC2  TM-Ca ({X.shape[1] // 3} atoms)<br>GMM k={MACRO_N_COMPONENTS}",
        axis_titles=(f"PC1  ({evr[0]:.1%} var)", f"PC2  ({evr[1]:.1%} var)"),
    )
    _reannotate(protein, meta, labels, x_col="pc_x", y_col="pc_y", method_tag=MACRO_METHOD_TAG,
                max_per_cluster=max_per_cluster)
    return MACRO_METHOD_TAG


# ── Ligand-pose clustering helpers (mirrors _notebook_setup_functions.py) ──

def _longest_chain_name(model):
    return max(model, key=lambda c: sum(1 for _ in c)).name


def _extract_ca(cif_path: Path):
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    chain_name = _longest_chain_name(model)
    coords, resids = [], []
    for residue in model[chain_name]:
        for atom in residue:
            if atom.name == "CA":
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                resids.append(residue.seqid.num)
    return np.array(coords, dtype=np.float32), np.array(resids, dtype=np.int32)


def _extract_ligand_atoms(cif_path: Path, ligand_chain: str):
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    coords, elements = [], []
    for residue in model[ligand_chain]:
        for atom in residue:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            elements.append(atom.element.name)
    return np.array(coords, dtype=np.float32), elements


def _resolved_ligand_info(protein: str):
    path = ABCFOLD_OUT_ROOT / f"{protein}__holo" / "abc_fold_input.resolved.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- {protein} has no holoform run (or it "
            "hasn't completed ABCfold yet), so there is no ligand to pose-cluster")
    data = json.loads(path.read_text())
    for seq in data["sequences"]:
        if "ligand" in seq:
            return seq["ligand"]["id"][0], seq["ligand"]["smiles"]
    raise ValueError(f"{path} has no 'ligand' entry in its sequences list")


def _tm_resids(protein: str):
    resids = pd.read_parquet(ALIGN_ROOT / f"{protein}__apo" / "resids.parquet")
    return resids.loc[resids["in_tm"], "resid"].to_numpy()


def _align_ensemble_iterative(frames, n_iter=5, tol=1e-4):
    ref = frames[0].copy()
    transforms = [(np.eye(3), np.zeros(3))] * len(frames)
    for _ in range(n_iter):
        transforms = [_kabsch_fit(frame, ref) for frame in frames]
        aligned = np.stack([(R @ frame.T).T + t for frame, (R, t) in zip(frames, transforms)])
        new_ref = aligned.mean(axis=0)
        shift = float(np.sqrt(((new_ref - ref) ** 2).sum(axis=1).mean()))
        ref = new_ref
        if shift < tol:
            break
    return ref, transforms


def _cluster_ligand_ensemble(cluster_rows, cif_by_key, ligand_chain, tm_resid_set, n_iter=5):
    align_frames, lig_frames, used_rows = [], [], []
    ref_n_ca, ref_elements = None, None
    n_skipped = {"cif": 0, "ca": 0, "ligand": 0}

    for _, row in cluster_rows.iterrows():
        cif = cif_by_key.get((row["source_run"], row["frame_id"]))
        if cif is None:
            n_skipped["cif"] += 1
            continue
        try:
            ca, resids = _extract_ca(cif)
            align_ca = ca[np.isin(resids, list(tm_resid_set))] if tm_resid_set is not None else ca
            lig_coords, elements = _extract_ligand_atoms(cif, ligand_chain)
        except Exception as e:
            print(f"[ligand-pca]   WARNING: failed to parse {cif} ({e}), skipping")
            n_skipped["cif"] += 1
            continue

        if ref_n_ca is None:
            ref_n_ca = len(align_ca)
        if len(align_ca) != ref_n_ca:
            n_skipped["ca"] += 1
            continue
        if ref_elements is None:
            ref_elements = elements
        if elements != ref_elements:
            n_skipped["ligand"] += 1
            continue

        align_frames.append(align_ca)
        lig_frames.append(lig_coords)
        used_rows.append(row)

    if any(n_skipped.values()):
        print(f"[ligand-pca]   skipped {n_skipped['cif']} (no CIF / parse failure), "
              f"{n_skipped['ca']} (Ca count mismatch), "
              f"{n_skipped['ligand']} (ligand atom mismatch) of {len(cluster_rows)} frames")

    if len(align_frames) < 2:
        return None, None

    _, transforms = _align_ensemble_iterative(align_frames, n_iter=n_iter)
    aligned_lig = [(R @ lig.T).T + t for lig, (R, t) in zip(lig_frames, transforms)]
    lig_X = np.stack(aligned_lig).reshape(len(aligned_lig), -1)
    return lig_X, pd.DataFrame(used_rows).reset_index(drop=True)


def _ca_cluster_assignments(protein: str, method_tag: str, iptm_threshold: float = IPTM_THRESHOLD):
    _, meta = _load_protein(protein, iptm_threshold=iptm_threshold)
    assign_path = REANN_ROOT / protein / method_tag / "assignments.parquet"
    if not assign_path.exists():
        raise FileNotFoundError(
            f"{assign_path} not found -- run cluster_macro_state({protein!r}) first")
    assignments = pd.read_parquet(assign_path)
    cluster_by_uid = (
        assignments.rename(columns={"frame_id": "unique_frame_id"})[["unique_frame_id", "cluster"]]
    )
    return meta.merge(cluster_by_uid, on="unique_frame_id", how="inner")


def _cluster_ligand_pose_for_ca_cluster(protein, cid, cluster_rows, cif_by_key, ligand_chain,
                                          tm_resid_set, method_tag, max_per_cluster,
                                          auto_k_max, n_iter=5):
    lig_X, used_meta = _cluster_ligand_ensemble(cluster_rows, cif_by_key, ligand_chain, tm_resid_set, n_iter=n_iter)
    if lig_X is None or len(used_meta) < MIN_HOLO_FRAMES:
        print(f"[ligand-pca] Ca-cluster {cid}: too few usable frames after "
              "CIF/Ca/ligand checks, skipping")
        return False

    n_lig_atoms = lig_X.shape[1] // 3
    n_pc = min(max(2, 5), lig_X.shape[1])
    pca = PCA(n_components=n_pc)
    coords = pca.fit_transform(lig_X)
    evr = pca.explained_variance_ratio_

    print(f"[ligand-pca] Ca-cluster {cid}: {len(used_meta)} holo frames, "
          f"{n_lig_atoms} ligand atoms, PC1={evr[0]:.1%} var"
          + (f", PC2={evr[1]:.1%} var" if len(evr) > 1 else ""))

    base_tag = f"{method_tag}_cluster{cid}_ligandpca"
    out_root = LIGPOSE_ROOT / protein / method_tag / f"ca_cluster_{cid}"

    if evr[0] > 0.95:
        pc1 = coords[:, 0]
        gmm, k_used, bic_by_k = _fit_gmm_bic_sweep(pc1.reshape(-1, 1), k_max=auto_k_max)
        method_title = f"Ca-cluster {cid}  Ligand-pose PCA  PC1-only fallback ({evr[0]:.1%} var > 95%)"
        _plot_bic_curve_svg(protein, method_title, bic_by_k, k_used, base_tag)

        labels = gmm.predict(pc1.reshape(-1, 1))
        used_meta = used_meta.copy()
        used_meta["lig_pc_x"] = pc1
        used_meta["gmm_cluster"] = labels
        hist_tag = f"{base_tag}_gmm_k{k_used}_hist1d"
        _plot_hist1d_svg(protein, used_meta, "lig_pc_x", k_used,
                          f"{protein}<br>{method_title}<br>GMM auto (BIC knee) k={k_used}, {n_lig_atoms} ligand atoms",
                          f"ligand PC1  ({evr[0]:.1%} var)", hist_tag)
        _reannotate(
            protein, used_meta, labels, x_col="lig_pc_x", y_col="lig_pc_x", method_tag=hist_tag,
            max_per_cluster=max_per_cluster, out_dir=out_root / hist_tag,
            table_description=LIGAND_POSE_TABLE_DESCRIPTION,
            column_descriptions=_ligand_pose_column_descriptions("lig_pc_x", "lig_pc_x"),
        )
        return True

    xy = coords[:, [0, 1]]
    gmm, k_used, bic_by_k = _fit_gmm_bic_sweep(xy, k_max=auto_k_max)
    method_title = f"Ca-cluster {cid}  Ligand-pose PCA"
    _plot_bic_curve_svg(protein, method_title, bic_by_k, k_used, base_tag)

    labels = gmm.predict(xy)
    used_meta = used_meta.copy()
    used_meta["lig_pc_x"], used_meta["lig_pc_y"] = xy[:, 0], xy[:, 1]
    used_meta["gmm_cluster"] = labels
    fig_tag = f"{base_tag}_gmm_k{k_used}"

    _plot_gmm_embedding_svg(
        protein, used_meta, xy, "lig_pc_x", "lig_pc_y", k_used, gmm, fig_tag,
        title=(f"{protein}  Ca-cluster {cid}<br>Ligand-pose PCA ({n_lig_atoms} atoms, "
               f"{len(used_meta)} holo frames)<br>{k_used} pose(s) (GMM)"),
        axis_titles=(f"lig PC1  ({evr[0]:.1%} var)", f"lig PC2  ({evr[1]:.1%} var)"),
    )
    _reannotate(
        protein, used_meta, labels, "lig_pc_x", "lig_pc_y", fig_tag,
        max_per_cluster=max_per_cluster, out_dir=out_root / fig_tag,
        table_description=LIGAND_POSE_TABLE_DESCRIPTION,
        column_descriptions=_ligand_pose_column_descriptions("lig_pc_x", "lig_pc_y"),
    )
    return True


def cluster_ligand_pose(protein: str, method_tag: str = MACRO_METHOD_TAG,
                         min_holo_frames: int = MIN_HOLO_FRAMES,
                         auto_k_max: int = LIGAND_AUTO_K_MAX,
                         max_per_cluster: int = 20, tm_only: bool = True, n_iter: int = 5) -> int:
    """GMM n_components="auto" ligand-pose sub-clustering within each
    macro-state cluster's holoform frames -- mirrors
    plot_ligand_pca(protein, method_tag, n_components="auto", auto_k_max=6).
    Returns the number of macro-state clusters that got a ligand-pose result."""
    merged = _ca_cluster_assignments(protein, method_tag)
    ligand_chain, smiles = _resolved_ligand_info(protein)
    cif_by_key = _build_cif_by_key(merged)
    tm_resid_set = set(int(r) for r in _tm_resids(protein)) if tm_only else None

    all_ids = sorted(int(c) for c in merged["cluster"].unique() if c != -1)
    print(f"[ligand-pca] {protein}: ligand chain {ligand_chain!r} ({smiles}), "
          f"{len(all_ids)} Ca cluster(s) to check (method_tag={method_tag!r}, "
          f"min_holo_frames={min_holo_frames}, tm_only={tm_only})")

    n_found = 0
    for cid in all_ids:
        cluster_rows = merged[(merged["cluster"] == cid) & (merged["status"] == "holo")]
        if len(cluster_rows) < min_holo_frames:
            print(f"[ligand-pca] Ca-cluster {cid}: {len(cluster_rows)} holo frame(s) "
                  f"< min_holo_frames={min_holo_frames}, skipping")
            continue
        ok = _cluster_ligand_pose_for_ca_cluster(
            protein, cid, cluster_rows, cif_by_key, ligand_chain, tm_resid_set,
            method_tag, max_per_cluster, auto_k_max, n_iter=n_iter,
        )
        n_found += int(ok)

    if n_found == 0:
        print(f"[ligand-pca] {protein}: no Ca cluster produced a ligand-pose result")
    return n_found


# ── Driver ──

def run(protein: str, max_per_cluster: int = 20) -> None:
    print(f"[cluster_conformations] {protein}: starting macro-state clustering")
    macro_tag = cluster_macro_state(protein, max_per_cluster=max_per_cluster)

    has_holo = (ABCFOLD_OUT_ROOT / f"{protein}__holo" / "abc_fold_input.resolved.json").exists()
    if has_holo and macro_tag == MACRO_METHOD_TAG:
        print(f"[cluster_conformations] {protein}: starting ligand-pose clustering")
        cluster_ligand_pose(protein, method_tag=macro_tag, max_per_cluster=max_per_cluster)
    elif has_holo:
        print(f"[cluster_conformations] {protein}: macro-state used tag {macro_tag!r} "
              f"(1-D fallback), not {MACRO_METHOD_TAG!r} -- skipping ligand-pose step")
    else:
        print(f"[cluster_conformations] {protein}: no holoform run -- skipping ligand-pose step")

    sentinel = REANN_ROOT / protein / MACRO_METHOD_TAG / "cluster.done"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(f"macro_method_tag={macro_tag}\nhas_holo={has_holo}\n")
    print(f"[cluster_conformations] {protein}: done -> {sentinel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein", required=True, help="BASE protein identifier, e.g. NPF2.12_Q9LFX9")
    ap.add_argument("--max-per-cluster", type=int, default=20,
                     help="cap on how many CIFs per cluster get symlinked for reannotation")
    args = ap.parse_args()
    run(args.protein, max_per_cluster=args.max_per_cluster)


if __name__ == "__main__":
    main()
