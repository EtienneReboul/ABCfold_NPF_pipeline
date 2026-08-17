from pathlib import Path

import re
import subprocess
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import Output, VBox
from IPython.display import display
import optuna
from hdbscan.validity import validity_index
from kneed import KneeLocator
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

optuna.logging.set_verbosity(optuna.logging.WARNING)  # one INFO line per trial is too noisy at hdbscan_n_trials=40/protein

ROOT             = Path("..")
sys.path.insert(0, str(ROOT / "scripts"))
from parquet_utils import write_parquet_with_metadata  # noqa: E402
ABCFOLD_OUT_ROOT = ROOT / "results" / "abcfold"
ALIGN_ROOT       = ROOT / "results" / "tm_alignment"
REANN_ROOT       = ROOT / "results" / "tm_reannotated"
FIG_ROOT         = ROOT / "results" / "figures"

GMM_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

# Fixed colours for the categorical color_by="status" scatter (apo vs holo)
STATUS_PALETTE = {"apo": "#7f7f7f", "holo": "#d62728"}

# Fixed colours for the categorical color_by="model" scatter -- the new
# axis this ABCfold pipeline exists for (AF3_NPF_pipeline's single-backend
# notebooks only ever had "status" to colour by; every protein here pools
# up to 6 backends' frames instead of one, see _load_protein below).
# "unknown" is _backend_of()'s fallback for a CIF path that doesn't match
# any BACKEND_PATTERNS entry -- shouldn't happen in practice, kept for
# safety since it's cheap to render if it ever does.
MODEL_PALETTE = {
    "alphafold3":   "#1f77b4",
    "boltz":        "#ff7f00",
    "chai1":        "#2ca02c",
    "openfold3":    "#d62728",
    "protenix":     "#9467bd",
    "rosettafold3": "#8c564b",
    "unknown":      "#7f7f7f",
}

# color_by name -> (palette dict, category display/legend order)
CATEGORICAL_COLOR_CONFIG = {
    "status": (STATUS_PALETTE, ["apo", "holo"]),
    "model":  (MODEL_PALETTE, ["alphafold3", "boltz", "chai1", "openfold3",
                               "protenix", "rosettafold3", "unknown"]),
}

# Ablation switch for plot_pca's `models` argument -- which ABCfold backends
# get pooled into the ensemble before the PCA fit. Flip a backend to False to
# ask e.g. "is AlphaFold3 still pulling its weight once OpenFold3 (open
# source, no EULA/weight-request form) is in the mix?" -- any subset can be
# toggled, not just AF3. Edit in place (`ENABLED_MODELS["alphafold3"] = False`)
# to change the default for every plot_pca call below, or pass a one-off
# `models={**ENABLED_MODELS, "alphafold3": False}` to a single call instead.
ENABLED_MODELS = {
    "alphafold3":   True,
    "boltz":        True,
    "chai1":        True,
    "openfold3":    True,
    "protenix":     True,
    "rosettafold3": True,
}

# Default HDBSCAN search space for cluster_method='hdbscan', n_clusters='auto'
# (Optuna/TPE + DBCV tuning) -- same candidate values/naming as
# NPF_pocket_pipeline/notebook/msa_clustering/all_proteins_blosum62_pca_hdbscan.ipynb
# and AF3_NPF_pipeline/notebook/tm_conformation_clustering_*.ipynb, applied
# to this project's 2-D embedding. "cityblock" not "manhattan": same
# distance, but that name errors inside hdbscan.validity.validity_index.
HDBSCAN_MIN_SAMPLES_CANDIDATES = [3, 5, 10, 15, 20, 25, 30]
HDBSCAN_MIN_CLUSTER_SIZE_CANDIDATES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
HDBSCAN_CLUSTER_SELECTION_METHODS = ["eom", "leaf"]
HDBSCAN_METRICS = ["euclidean", "cityblock"]

# color_by name -> (meta column, colorscale, (cmin, cmax) or None for data-range)
CONTINUOUS_COLOR_CONFIG = {
    "ptm":     ("ptm",     "Viridis", (0.0, 1.0)),
    "iptm":    ("iptm",    "Viridis", (0.0, 1.0)),  # NaN on AF3 apoform frames (no interface to score); 0.0 for the same case on the other 5 backends
    "seed":    ("seed",    "Turbo",   None),
    "rmsd_tm": ("rmsd_tm", "Plasma",  None),  # per-frame RMSD (A) to this ensemble's converged TM-helix mean
}

HOVER_COLS = ["unique_frame_id", "status", "model", "seed", "sample_index", "ptm", "iptm"]
HOVER_TEMPLATE_BODY = (
    "%{customdata[0]}<br>"
    "status: %{customdata[1]}  ·  model: %{customdata[2]}<br>"
    "seed %{customdata[3]}  ·  sample %{customdata[4]}<br>"
    "pTM: %{customdata[5]:.3f}  ·  ipTM: %{customdata[6]:.3f}<br>"
)


def _save_fig(fig, protein, filename):
    """Write a static PNG copy of fig under results/figures/<protein>/ (via
    kaleido) so plots survive a `results/` -> `results_vN/` rename instead
    of only living in the notebook's cell output / plotly's interactive
    fig.show()."""
    out_dir = FIG_ROOT / protein
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.write_image(str(path), scale=2)
    return path


def _load_run(run_name):
    """Load one apo/holo run's aligned TM-Ca ensemble + per-frame metadata
    (model/backend, seed, sample index, pTM, RMSD-to-mean), written by
    scripts/tm_helix_alignment.py -- already pooled across every one of
    ABCfold's 6 backends (AlphaFold3, Boltz-2, Chai-1, OpenFold3, Protenix,
    RosettaFold3) x seed x diffusion/sample for this run, unlike
    AF3_NPF_pipeline's equivalent (AF3 only). `run_name` is a full
    apo/holo run identifier (e.g. 'NPF2.12_Q9LFX9__apo'), matching a
    results/tm_alignment/<run_name>/ directory."""
    npy = ALIGN_ROOT / run_name / "aligned_ca_tm.npy"
    meta_path = ALIGN_ROOT / run_name / "meta.parquet"
    if not npy.exists():
        raise FileNotFoundError(
            f"{npy} not found -- run worflows/postprocessing/Snakefile "
            f"(scripts/tm_helix_alignment.py) for {run_name} first")
    coords = np.load(npy)                          # (n_frames, n_ca_tm, 3)
    meta   = pd.read_parquet(meta_path)
    X      = coords.reshape(coords.shape[0], -1)   # flatten to (n_frames, n_ca_tm*3)
    return X, meta


def _kabsch_fit(P, Q):
    """Rotation R (3,3) and translation t (3,) such that (R @ P.T).T + t ~= Q.
    Same as kabsch() in scripts/tm_helix_alignment.py."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_mean - R @ p_mean
    return R, t


def _load_protein(protein):
    """Load the aligned, multi-backend TM-Ca ensemble for one BASE protein
    (e.g. 'NPF2.12_Q9LFX9'), merging its apoform and holoform ABCfold runs
    into a single pooled ensemble for analysis. Apoform always exists (or
    is expected to); holoform only when worflows/preprocessing/Snakefile's
    ligand_for() assigned a ligand AND that run has completed -- if
    there's no holoform run yet, only the apoform ensemble is returned.

    scripts/tm_helix_alignment.py's align_ensemble() converges each run
    (pooling all 6 backends already) to ITS OWN ensemble mean,
    independently -- apo and holo are separate ABCfold jobs with no shared
    global orientation, so their two reference frames can differ by an
    arbitrary rigid-body rotation/translation. Pooling them naively would
    make the PCA fit pick up that arbitrary offset instead of real
    ligand-induced conformational shifts, so holo's mean TM structure is
    Kabsch-refit onto apo's mean TM structure here (apo is the anchor
    since it always exists) and that one rigid-body transform is applied
    to every holo frame before pooling -- this only corrects a
    whole-ensemble offset, not per-frame noise, so a single fit is correct
    and sufficient.

    Adds three columns to meta: 'status' ('apo'/'holo'), 'source_run' (the
    underlying results/tm_alignment/<source_run>/ and
    results/abcfold/<source_run>/ directory name, needed by _reannotate to
    find the right CIFs), and 'unique_frame_id' (frame_id prefixed with
    status) since apo and holo runs independently repeat the same
    model/seed/sample numbering and would otherwise collide once pooled.
    'model' (the backend: alphafold3/boltz/chai1/openfold3/protenix/
    rosettafold3) is already a meta.parquet column written by
    scripts/tm_helix_alignment.py -- untouched here, just carried through.
    """
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
        coords = X.reshape(X.shape[0], -1, 3)  # (n_frames, n_ca_tm, 3)

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

    X    = np.concatenate(X_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, meta


# Mirrors scripts/tm_helix_alignment.py's BACKEND_PATTERNS / backend_of() /
# discover_predictions() / parse_frame_id() -- duplicated here (not
# imported) so this notebook stays self-contained, same convention as
# _kabsch_fit above mirroring that script's kabsch(). Keep in sync if the
# script's version changes. Only the frame_id half of parse_frame_id is
# needed here (to match a rediscovered CIF back to its meta.parquet row by
# (source_run, frame_id) for reannotation) -- model/seed/sample_index are
# already columns tm_helix_alignment.py wrote to meta.parquet itself.

BACKEND_PATTERNS = {
    "alphafold3":   "alphafold3",
    "boltz":        "boltz",
    "chai1":        "chai",
    "openfold3":    "openfold",
    "protenix":     "protenix",
    "rosettafold3": "rosettafold",
}


def _backend_of(path, predictions_dir):
    try:
        top = path.relative_to(predictions_dir).parts[0].lower()
    except (ValueError, IndexError):
        return "unknown"
    for backend, pattern in BACKEND_PATTERNS.items():
        if pattern in top:
            return backend
    return "unknown"


def _discover_abcfold_cifs(run_name):
    """Every model CIF ABCfold produced for one apo/holo run, pooled across
    all 6 backends x seed x diffusion/sample. Mirrors
    scripts/tm_helix_alignment.py's discover_predictions()."""
    predictions_dir = ABCFOLD_OUT_ROOT / run_name
    return sorted(c for c in predictions_dir.rglob("*.cif") if "templates" not in c.parts)


def _frame_id_for_cif(cif_path, predictions_dir):
    """Same frame_id derivation as scripts/tm_helix_alignment.py's
    parse_frame_id(), so a rediscovered CIF can be matched back to its
    meta.parquet row by (source_run, frame_id)."""
    rel   = cif_path.relative_to(predictions_dir)
    model = _backend_of(cif_path, predictions_dir)
    m = re.search(r"seed-?(\d+)_sample-?(\d+)", str(rel), re.IGNORECASE)
    if m:
        return f"{model}_seed{m.group(1)}_sample{m.group(2)}"
    return f"{model}_{rel.with_suffix('')}".replace("/", "_")


def _build_cif_by_key(meta):
    """(source_run, frame_id) -> resolved CIF Path, for every source_run in
    this (possibly apo+holo-pooled) meta. Shared by _reannotate (bulk
    symlinking) and _make_reveal_handler (click-to-reveal-in-Finder)."""
    cif_by_key = {}
    for source_run in sorted(meta["source_run"].unique()):
        predictions_dir = ABCFOLD_OUT_ROOT / source_run
        for c in _discover_abcfold_cifs(source_run):
            cif_by_key[(source_run, _frame_id_for_cif(c, predictions_dir))] = c
    return cif_by_key


def _ellipse_trace(mean, cov, color, n_std=1.5, n_pts=80):
    vals, vecs = np.linalg.eigh(cov)
    idx        = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    t          = np.linspace(0, 2 * np.pi, n_pts)
    pts        = n_std * (vecs * np.sqrt(np.maximum(vals, 0))) @ np.vstack(
                     [np.cos(t), np.sin(t)])
    x, y = mean[0] + pts[0], mean[1] + pts[1]
    return go.Scatter(
        x=np.append(x, x[0]), y=np.append(y, y[0]),
        mode="lines",
        line=dict(color=color, width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    )


ASSIGNMENTS_TABLE_DESCRIPTION = (
    "Cluster membership for one protein's conformational ensemble (GMM or "
    "HDBSCAN -- method_tag distinguishes which), written by _reannotate(). One "
    "row per pooled apo+holo frame; symlinked marks whether that frame's CIF "
    "was actually symlinked into this directory's cluster_<k>/ (frames beyond "
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
    "cluster": "Cluster id this frame was assigned to (-1 = HDBSCAN noise; GMM never produces -1)",
    "symlinked": "True if this frame's CIF was actually symlinked into cluster_<k>/ "
                 "(capped at max_per_cluster per cluster -- every frame is still listed "
                 "here regardless)",
}


def _assignments_column_descriptions(x_col, y_col):
    """ASSIGNMENTS_FIXED_COLUMN_DESCRIPTIONS plus the two embedding-coordinate
    columns, which are named dynamically (always 'pc_x'/'pc_y' in practice --
    see _plot_embedding/_plot_pca_1d_fallback -- but _reannotate's signature
    is generic, so this stays generic too)."""
    coord_desc = ("2D embedding coordinate this clustering was run on -- a PCA "
                  "component for the 2-D path, or the 1-D PCA value duplicated "
                  "onto both axes for _plot_pca_1d_fallback's path")
    return {
        **ASSIGNMENTS_FIXED_COLUMN_DESCRIPTIONS,
        x_col: coord_desc + " (x axis)",
        y_col: coord_desc + " (y axis)",
    }


def _reannotate(protein, meta, labels, x_col, y_col, method_tag,
                max_per_cluster=20, sample_seed=42):
    """Symlink each structure's CIF into results/tm_reannotated/<protein>/<method_tag>/cluster_<k>/.

    `protein` is a BASE protein name; `meta` (from the merged
    _load_protein) covers both its apo and holo ABCfold runs, each under
    its own results/abcfold/<source_run>/ directory and pooling up to 6
    backends, so CIFs are looked up by (source_run, frame_id) rather than
    frame_id alone -- apo and holo runs independently repeat the same
    model/seed/sample numbering, so frame_id on its own is ambiguous once
    pooled. Looked up by frame_id (written into meta.parquet by
    scripts/tm_helix_alignment.py) rather than positional zip, since that
    script may have skipped a frame mid-ensemble (Ca count mismatch) so a
    fresh CIF glob need not line up index-for-index with meta.parquet.
    Symlinks are named "<unique_frame_id>.cif" (frame_id, itself already
    backend-prefixed by parse_frame_id, prefixed again with status) so
    apo/holo filenames never collide once multiple frames land in the same
    cluster_dir.

    Clusters routinely hold far more structures than is useful to load into
    ChimeraX at once, so at most `max_per_cluster` structures per cluster
    are randomly subsampled (without replacement, `sample_seed` for
    reproducibility) and only those get symlinked to disk. `assignments.parquet`
    still lists every frame in the cluster (with a `symlinked` column) so
    the full membership stays available for downstream stats even though
    the on-disk CIF set is capped.

    Cluster ids are read from the data (`sorted(set(labels))`) rather than
    assumed to be `range(0, labels.max() + 1)`, so HDBSCAN's `-1` noise
    label gets its own `cluster_noise/` directory instead of being silently
    dropped (GMM labels are always a contiguous 0..k-1 range, so this is a
    no-op for the GMM path).
    """
    cif_by_key = _build_cif_by_key(meta)

    meta = meta.copy()
    meta["gmm_cluster"] = labels
    out_dir     = REANN_ROOT / protein / method_tag
    cluster_ids = sorted(set(int(l) for l in labels))

    assign_rows = []
    n_symlinked = 0
    for cid in cluster_ids:
        dir_name    = "cluster_noise" if cid == -1 else f"cluster_{cid}"
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
                # RosettaFold3 writes both a "_model.cif" and a
                # "_model_fixed.cif" per (seed, sample) -- near-identical
                # coordinates, but two distinct source CIFs that collide on
                # the same frame_id (parse_frame_id's regex only captures
                # seed/sample, not the "_fixed" suffix), so two meta.parquet
                # rows can legitimately share one unique_frame_id. Skip
                # rather than crash on the second one; the first symlink
                # already represents this frame_id in this cluster_dir.
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
        table_description=ASSIGNMENTS_TABLE_DESCRIPTION,
        column_descriptions=_assignments_column_descriptions(x_col, y_col),
    )
    print(f"[reannotate] {protein}/{method_tag}: {n_symlinked} symlinks "
          f"(max {max_per_cluster}/cluster) of {len(assign_rows)} assignments -> {out_dir}")


def _fit_gmm_bic_sweep(xy, k_min=1, k_max=20, n_init=20, random_state=42):
    """Fit a GaussianMixture for every k in [k_min, k_max] on the 2-D embedding
    and return the model sitting at the knee of the BIC-vs-k curve.

    Mirrors find_best_k in NPF_pocket_pipeline/scripts/gmm_conformation.py:
    KneeLocator's default interp1d interpolation follows every point of the
    BIC curve exactly, so a single noisy value (e.g. a bad n_init restart)
    reads as a spurious knee right at the first bump. Fitting a polynomial
    through the curve first (interp_method="polynomial", degree capped
    relative to the number of k's swept) smooths that out and finds the
    real elbow instead. Falls back to the raw BIC minimum if KneeLocator
    finds no knee.
    """
    k_max = min(k_max, xy.shape[0] - 1)
    ks    = list(range(max(1, k_min), k_max + 1))

    gmms, bic_by_k = {}, {}
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=n_init, random_state=random_state)
        try:
            gmm.fit(xy)
        except ValueError as e:
            # A component can collapse onto too few/duplicate points for a
            # given k (ill-defined covariance) on some ensembles -- same
            # per-candidate catch-and-skip _fit_hdbscan_dbcv_search already
            # does below, applied here so one bad k doesn't crash the whole
            # sweep (confirmed on a real run: NPF2.2_Q9M174, apoform-only).
            print(f"[gmm-auto] WARNING: k={k} failed ({e}), skipping")
            continue
        gmms[k]     = gmm
        bic_by_k[k] = float(gmm.bic(xy))

    if not bic_by_k:
        raise RuntimeError(
            f"GMM auto (BIC sweep) failed for every k in [{ks[0]}, {ks[-1]}] -- "
            "try a narrower auto_k_min/auto_k_max range or n_clusters=<int> (manual)")
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


def _plot_bic_curve(protein, method_title, bic_by_k, best_k, method_tag):
    ks   = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=bics, mode="lines+markers",
        line=dict(color="#1565C0", width=2), marker=dict(size=6),
        name="BIC",
    ))
    fig.add_trace(go.Scatter(
        x=[best_k], y=[bic_by_k[best_k]], mode="markers",
        marker=dict(size=14, color="#d62728", symbol="star"),
        name=f"knee k={best_k}",
    ))
    fig.update_layout(
        title=f"{protein}<br>{method_title}<br>BIC sweep k={ks[0]}-{ks[-1]}, knee k={best_k}",
        xaxis_title="n_components (k)", yaxis_title="BIC",
        template="plotly_white", height=380, width=520, showlegend=False,
    )
    _save_fig(fig, protein, f"{method_tag}_bic_k{best_k}.png")
    fig.show()


def _fit_hdbscan_dbcv_search(xy, min_samples_candidates=HDBSCAN_MIN_SAMPLES_CANDIDATES,
                              min_cluster_size_candidates=HDBSCAN_MIN_CLUSTER_SIZE_CANDIDATES,
                              cluster_selection_methods=HDBSCAN_CLUSTER_SELECTION_METHODS,
                              metrics=HDBSCAN_METRICS, n_trials=400, random_state=42):
    """Optuna/TPE search over (min_samples, min_cluster_size, cluster_selection_method,
    metric) for HDBSCAN on the 2-D embedding, scored by DBCV (Moulavi et al. 2014) via
    hdbscan.validity.validity_index -- same approach as
    select_hdbscan_hyperparams_and_cluster in
    NPF_pocket_pipeline/notebook/msa_clustering/all_proteins_blosum62_pca_hdbscan.ipynb,
    just applied to a 2-D embedding here instead of a full BLOSUM62-encoded
    sequence embedding. TPE models which regions of the search space tend to
    score well on DBCV as trials complete and concentrates later trials
    there, rather than sampling the grid uniformly at random. Combos
    yielding fewer than 2 clusters, or that error inside DBCV, score -1.0
    so they're never selected.
    """
    n = xy.shape[0]
    max_min_cluster_size = max(2, n // 5)
    candidate_min_cluster_sizes = [m for m in min_cluster_size_candidates if 2 <= m <= max_min_cluster_size]
    if not candidate_min_cluster_sizes:
        candidate_min_cluster_sizes = [max_min_cluster_size]

    grid_size = (len(min_samples_candidates) * len(candidate_min_cluster_sizes)
                 * len(cluster_selection_methods) * len(metrics))
    n_trials = min(n_trials, grid_size)

    def objective(trial):
        min_samples = trial.suggest_categorical("min_samples", list(min_samples_candidates))
        min_cluster_size = trial.suggest_categorical("min_cluster_size", candidate_min_cluster_sizes)
        cluster_selection_method = trial.suggest_categorical("cluster_selection_method", list(cluster_selection_methods))
        metric = trial.suggest_categorical("metric", list(metrics))
        try:
            labels = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,
                              cluster_selection_method=cluster_selection_method,
                              metric=metric, copy=False).fit(xy).labels_
            n_clust = len(set(c for c in labels if c >= 0))
            dbcv = float(validity_index(xy.astype(np.float64), labels, metric=metric)) if n_clust >= 2 else -1.0
        except Exception as e:
            print(f"[hdbscan-auto] combo ms={min_samples} mcs={min_cluster_size} "
                  f"{cluster_selection_method}/{metric} failed: {e}")
            labels, dbcv = None, -1.0
        trial.set_user_attr("labels", None if labels is None else labels.tolist())
        return dbcv

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial  = study.best_trial
    best_labels = best_trial.user_attrs["labels"]
    best = {
        "min_samples": best_trial.params["min_samples"],
        "min_cluster_size": best_trial.params["min_cluster_size"],
        "cluster_selection_method": best_trial.params["cluster_selection_method"],
        "metric": best_trial.params["metric"],
        "dbcv": best_trial.value,
    }
    labels = np.array(best_labels) if best_labels is not None else np.full(n, -1)
    return labels, best, study


def _plot_dbcv_search(protein, method_title, study, best, method_tag):
    dbcvs = sorted(study.trials_dataframe()["value"].fillna(-1.0).tolist(), reverse=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(dbcvs))), y=dbcvs, marker_color="#1565C0", name="DBCV"))
    fig.add_hline(y=best["dbcv"], line_dash="dash", line_color="#d62728",
                  annotation_text=f"best DBCV={best['dbcv']:.3f}")
    fig.update_layout(
        title=f"{protein}<br>{method_title}<br>HDBSCAN Optuna/TPE search, {len(dbcvs)} trials",
        xaxis_title="trial (sorted by DBCV)", yaxis_title="DBCV",
        template="plotly_white", height=380, width=520, showlegend=False,
    )
    _save_fig(fig, protein, f"{method_tag}_hdbscan_dbcv_search.png")
    fig.show()


def _make_reveal_handler(sub, cif_by_key, out):
    """Click-to-reveal-in-Finder handler for one FigureWidget trace. `sub`
    is the exact per-trace subset DataFrame _plot_embedding built that
    trace's x/y from (categorical category, GMM/HDBSCAN cluster, noise, or
    continuous-color subset) -- `points.point_inds` are positions INTO that
    subset, in the same order, so `sub.iloc[idx]` recovers the right row.
    Resolves (source_run, frame_id) -> actual CIF via cif_by_key (same
    lookup _reannotate uses for symlinking) and shells out to `open -R`
    (macOS-only: reveals the file, highlighted, in Finder)."""
    def _on_click(trace, points, state):
        with out:
            for idx in points.point_inds:
                row = sub.iloc[idx]
                cif = cif_by_key.get((row["source_run"], row["frame_id"]))
                if cif is None:
                    print(f"[reveal] no CIF found for {row['unique_frame_id']} ({row['source_run']})")
                    continue
                print(f"[reveal] {row['unique_frame_id']}  ({row['model']}, pTM={row['ptm']:.3f})  -> {cif}")
                subprocess.run(["open", "-R", str(cif)])
    return _on_click


def _plot_embedding(protein, meta, xy, x_col, y_col, method_tag, method_title,
                     color_by="model", cluster_method="gmm", n_clusters=None,
                     max_per_cluster=20, auto_k_min=1, auto_k_max=20,
                     hdbscan_min_cluster_size=None, hdbscan_min_samples=None,
                     hdbscan_cluster_selection_method="eom", hdbscan_metric="euclidean",
                     hdbscan_n_trials=40,
                     marker_size=6, opacity=0.7, axis_titles=("dim 1", "dim 2")):
    """Shared scatter / cluster / reannotate renderer for PCA.

    color_by is a CATEGORICAL_COLOR_CONFIG key ("model" default -- which of
    the 6 ABCfold backends produced each frame, see MODEL_PALETTE; or
    "status" -- apo/holo) or a CONTINUOUS_COLOR_CONFIG key ("ptm", "iptm", "seed",
    or "rmsd_tm"). n_clusters, if set, fits cluster_method ("gmm" default,
    or "hdbscan") on the 2-D embedding instead and colours by cluster:

    - cluster_method="gmm": an int n_clusters fits exactly that many GMM
      components (manual); "auto" sweeps auto_k_min..auto_k_max components
      and picks the BIC-curve knee instead (see _fit_gmm_bic_sweep).
    - cluster_method="hdbscan": n_clusters="auto" searches
      (hdbscan_min_cluster_size, hdbscan_min_samples,
      hdbscan_cluster_selection_method, hdbscan_metric) with Optuna/TPE
      scored by DBCV (see _fit_hdbscan_dbcv_search, hdbscan_n_trials
      trials); "manual" fits HDBSCAN directly with the explicit hdbscan_*
      arguments (hdbscan_min_cluster_size is required in that case). Points
      HDBSCAN calls noise (-1) are shown as unclustered ("x" markers)
      rather than being assigned a colour.

    max_per_cluster caps how many of each cluster's CIFs get symlinked for
    reannotation (see _reannotate); HDBSCAN's noise points get their own
    cluster_noise/ subsample rather than being dropped.

    Every figure this function produces (the BIC/DBCV diagnostic plot, when
    applicable, and the main embedding scatter) is also written as a static
    PNG under results/figures/<protein>/ via _save_fig, tagged with the same
    method/cluster identifier used for reannotation symlinks -- so plots
    survive a `results/` -> `results_vN/` rename instead of only existing as
    notebook cell output.
    """
    meta = meta.copy()
    meta[x_col], meta[y_col] = xy[:, 0], xy[:, 1]

    fig = go.FigureWidget()  # FigureWidget (not Figure): keeps a live comm channel to this kernel so
                              # on_click below can run local Python (open -R) when you click a point
    cif_by_key = _build_cif_by_key(meta)
    out = Output()  # captures click-feedback prints -- a callback fired via the comm channel doesn't
                     # reliably print to any visible cell on its own, so route it through this instead

    if n_clusters is not None and cluster_method == "gmm":
        # -- GMM clustering on the 2-D embedding (manual k or auto BIC-knee) --
        if n_clusters == "auto":
            gmm, k_used, bic_by_k = _fit_gmm_bic_sweep(xy, k_min=auto_k_min, k_max=auto_k_max)
            _plot_bic_curve(protein, method_title, bic_by_k, k_used, method_tag)
            cluster_label = f"GMM auto (BIC knee) k={k_used}"
        else:
            k_used = n_clusters
            gmm = GaussianMixture(n_components=k_used, covariance_type="full",
                                   n_init=20, random_state=42)
            gmm.fit(xy)
            cluster_label = f"GMM k={k_used}"

        labels = gmm.predict(xy)
        meta["gmm_cluster"] = labels
        fig_tag = f"{method_tag}_k{k_used}"

        cdata = meta[HOVER_COLS].fillna("?").values
        for k in range(k_used):
            sub   = meta[meta["gmm_cluster"] == k]
            color = GMM_PALETTE[k % len(GMM_PALETTE)]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=f"cluster {k}",
                customdata=cdata[meta["gmm_cluster"].values == k],
                hovertemplate=(
                    HOVER_TEMPLATE_BODY +
                    f"{x_col}: %{{x:.3f}}  {y_col}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ))
            fig.data[-1].on_click(_make_reveal_handler(sub, cif_by_key, out))
            fig.add_trace(_ellipse_trace(gmm.means_[k], gmm.covariances_[k], color))

        legend_title = "GMM cluster"
        _reannotate(protein, meta, labels, x_col, y_col, fig_tag,
                    max_per_cluster=max_per_cluster)

    elif n_clusters is not None and cluster_method == "hdbscan":
        # -- HDBSCAN clustering on the 2-D embedding (auto DBCV-tuned or manual) --
        if n_clusters == "auto":
            labels, best, study = _fit_hdbscan_dbcv_search(xy, n_trials=hdbscan_n_trials)
            _plot_dbcv_search(protein, method_title, study, best, method_tag)
            cluster_label = (f"HDBSCAN auto (DBCV={best['dbcv']:.3f}) "
                              f"mcs={best['min_cluster_size']} ms={best['min_samples']} "
                              f"{best['cluster_selection_method']}/{best['metric']}")
            fig_tag = f"{method_tag}_hdbscan_auto"
        elif n_clusters == "manual":
            if hdbscan_min_cluster_size is None:
                raise ValueError(
                    "cluster_method='hdbscan' with n_clusters='manual' requires "
                    "hdbscan_min_cluster_size to be set")
            labels = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                              min_samples=hdbscan_min_samples,
                              cluster_selection_method=hdbscan_cluster_selection_method,
                              metric=hdbscan_metric).fit(xy).labels_
            cluster_label = (f"HDBSCAN manual mcs={hdbscan_min_cluster_size} "
                              f"ms={hdbscan_min_samples} "
                              f"{hdbscan_cluster_selection_method}/{hdbscan_metric}")
            fig_tag = f"{method_tag}_hdbscan_manual_mcs{hdbscan_min_cluster_size}"
        else:
            raise ValueError(
                "cluster_method='hdbscan' requires n_clusters='auto' or 'manual' "
                f"(got {n_clusters!r})")

        meta["gmm_cluster"] = labels
        cdata = meta[HOVER_COLS].fillna("?").values
        cluster_ids = sorted(c for c in set(labels) if c >= 0)
        for i, k in enumerate(cluster_ids):
            sub   = meta[meta["gmm_cluster"] == k]
            color = GMM_PALETTE[i % len(GMM_PALETTE)]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=f"cluster {k}",
                customdata=cdata[meta["gmm_cluster"].values == k],
                hovertemplate=(
                    HOVER_TEMPLATE_BODY +
                    f"{x_col}: %{{x:.3f}}  {y_col}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ))
            fig.data[-1].on_click(_make_reveal_handler(sub, cif_by_key, out))

        noise = meta[meta["gmm_cluster"] == -1]
        if not noise.empty:
            fig.add_trace(go.Scatter(
                x=noise[x_col], y=noise[y_col],
                mode="markers",
                marker=dict(size=marker_size - 1, color="#aaa", opacity=0.4, symbol="x"),
                name="noise (HDBSCAN)",
                customdata=noise[HOVER_COLS].fillna("?").values,
                hovertemplate=(
                    HOVER_TEMPLATE_BODY +
                    f"{x_col}: %{{x:.3f}}  {y_col}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ))
            fig.data[-1].on_click(_make_reveal_handler(noise, cif_by_key, out))

        legend_title = "HDBSCAN cluster"
        _reannotate(protein, meta, labels, x_col, y_col, fig_tag,
                    max_per_cluster=max_per_cluster)

    elif color_by in CATEGORICAL_COLOR_CONFIG:
        # -- categorical colour scale: "model" (default, 6 ABCfold backends) or "status" --
        palette, order = CATEGORICAL_COLOR_CONFIG[color_by]
        present = set(meta[color_by].dropna())
        for category in [c for c in order if c in present]:
            sub = meta[meta[color_by] == category]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode="markers",
                marker=dict(size=marker_size, color=palette.get(category, "#7f7f7f"), opacity=opacity),
                name=category,
                customdata=sub[HOVER_COLS].fillna("?").values,
                hovertemplate=(
                    HOVER_TEMPLATE_BODY +
                    f"{x_col}: %{{x:.3f}}  {y_col}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ))
            fig.data[-1].on_click(_make_reveal_handler(sub, cif_by_key, out))

        cluster_label = f"color_by={color_by}"
        legend_title  = color_by
        fig_tag = f"{method_tag}_color_{color_by}"

    else:
        # -- continuous colour scale (pTM, seed, or rmsd_tm) --
        if color_by not in CONTINUOUS_COLOR_CONFIG:
            raise ValueError(f"Unknown color_by {color_by!r} -- expected one of "
                              f"{list(CATEGORICAL_COLOR_CONFIG)} or {list(CONTINUOUS_COLOR_CONFIG)}")
        col, colorscale, bounds = CONTINUOUS_COLOR_CONFIG[color_by]
        has_val = meta[col].notna()
        if not has_val.any():
            raise ValueError(
                f"color_by={color_by!r} has no non-NaN values anywhere in this ensemble "
                f"(e.g. find_confidence() in scripts/tm_helix_alignment.py found no pTM "
                f"for any backend here) -- try color_by='model' or 'status' instead")
        sub, missing = meta[has_val], meta[~has_val]
        cmin, cmax = bounds if bounds is not None else (sub[col].min(), sub[col].max())
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            mode="markers",
            marker=dict(size=marker_size, color=sub[col], colorscale=colorscale,
                        cmin=cmin, cmax=cmax, opacity=opacity,
                        colorbar=dict(title=color_by)),
            customdata=sub[HOVER_COLS].fillna("?").values,
            hovertemplate=(
                HOVER_TEMPLATE_BODY +
                f"{x_col}: %{{x:.3f}}  {y_col}: %{{y:.3f}}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.data[-1].on_click(_make_reveal_handler(sub, cif_by_key, out))
        if not missing.empty:
            fig.add_trace(go.Scatter(
                x=missing[x_col], y=missing[y_col],
                mode="markers",
                marker=dict(size=marker_size - 1, color="#aaa",
                            opacity=0.4, symbol="x"),
                name=f"{color_by} missing",
            ))
            fig.data[-1].on_click(_make_reveal_handler(missing, cif_by_key, out))

        cluster_label = f"color_by={color_by}"
        legend_title  = color_by
        fig_tag = f"{method_tag}_color_{color_by}"

    fig.update_layout(
        title=f"{protein}<br>{method_title}<br>{cluster_label}",
        xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
        legend_title=legend_title,
        template="plotly_white",
        height=540, width=680,
        hovermode="closest",
        hoverlabel=dict(font_size=11, namelength=0),
        hoverdistance=30,
    )
    _save_fig(fig, protein, f"{fig_tag}_embedding.png")
    display(VBox([fig, out]))


def _plot_pca_1d_fallback(protein, meta, pc1, evr1, method_tag, x_col="pc_x",
                           max_per_cluster=20, auto_k_min=1, auto_k_max=20):
    """Histogram + 1-D GMM auto (BIC-knee) clustering fallback for plot_pca
    when PC1 alone already explains >95% of the variance (see fallback_1d
    on plot_pca) -- at that point PC2 is close to pure noise, so the usual
    2-D scatter is misleading; a 1-D histogram of PC1, split by GMM
    cluster, is the honest representation instead. Always GMM auto
    (BIC-knee sweep, see _fit_gmm_bic_sweep) regardless of plot_pca's own
    cluster_method/n_components arguments -- HDBSCAN's tuning knobs don't
    carry over to a single dimension.
    """
    meta = meta.copy()
    meta[x_col] = pc1

    gmm, k_used, bic_by_k = _fit_gmm_bic_sweep(pc1.reshape(-1, 1), k_min=auto_k_min, k_max=auto_k_max)
    method_title = f"PCA  PC1-only fallback ({evr1:.1%} var > 95%)"
    _plot_bic_curve(protein, method_title, bic_by_k, k_used, method_tag)

    labels = gmm.predict(pc1.reshape(-1, 1))
    meta["gmm_cluster"] = labels
    fig_tag = f"{method_tag}_k{k_used}_hist1d"

    fig = go.Figure()
    for k in range(k_used):
        sub   = meta[meta["gmm_cluster"] == k]
        color = GMM_PALETTE[k % len(GMM_PALETTE)]
        fig.add_trace(go.Histogram(
            x=sub[x_col], name=f"cluster {k}", marker_color=color, opacity=0.7,
        ))
    fig.update_layout(
        title=f"{protein}<br>{method_title}<br>GMM auto (BIC knee) k={k_used}",
        xaxis_title=f"PC1  ({evr1:.1%} var)", yaxis_title="count",
        barmode="overlay", legend_title="GMM cluster",
        template="plotly_white", height=430, width=680,
    )
    _save_fig(fig, protein, f"{fig_tag}_hist.png")
    fig.show()

    _reannotate(protein, meta, labels, x_col=x_col, y_col=x_col, method_tag=fig_tag,
                max_per_cluster=max_per_cluster)


def plot_pca(protein: str, models: dict = None, color_by: str = "model", cluster_method: str = "gmm", n_components=None,
             max_per_cluster: int = 20, auto_k_min: int = 1, auto_k_max: int = 20,
             hdbscan_min_cluster_size=None, hdbscan_min_samples=None,
             hdbscan_cluster_selection_method: str = "eom", hdbscan_metric: str = "euclidean",
             hdbscan_n_trials: int = 40,
             pc_x: int = 1, pc_y: int = 2,
             marker_size: int = 6, opacity: float = 0.7,
             fallback_1d: bool = True):
    """PCA scatter of the aligned, multi-backend TM-Ca ensemble for one BASE
    protein, apo and holo runs pooled together (see _load_protein).

    Parameters
    ----------
    protein       BASE protein identifier, e.g. "NPF2.12_Q9LFX9" -- its
                  apoform and holoform ABCfold runs are merged into one
                  ensemble before plotting (see _load_protein), each run
                  itself already pooling up to 6 backends x every seed x
                  diffusion/sample (scripts/tm_helix_alignment.py).
    models        ablation switch -- dict of backend name -> True/False
                  (BACKEND_PATTERNS keys: alphafold3/boltz/chai1/openfold3/
                  protenix/rosettafold3), only frames from backends mapped to
                  True are pooled into the PCA fit. Defaults to the module-level
                  ENABLED_MODELS (edit that dict to change every call's default,
                  or pass a one-off dict here, e.g.
                  models={**ENABLED_MODELS, "alphafold3": False}, to ask
                  whether a given backend's conformations are already
                  recovered by the others without it).
    color_by      "model" (default -- categorical, which of the 6 ABCfold
                  backends produced each frame, see MODEL_PALETTE -- the
                  key new axis this pipeline exists for, since a single
                  backend's diffusion doesn't always recover every
                  conformation a second one finds), "status" (categorical
                  apo/holo, see STATUS_PALETTE), "ptm"/"iptm" (continuous 0-1
                  predicted TM-score confidence), "seed" (continuous,
                  data-range colour scale -- spot outlier seeds), or
                  "rmsd_tm" (continuous, per-frame RMSD in A to this
                  ensemble's converged TM-helix mean). Ignored when
                  n_components is set, or when the fallback_1d histogram
                  fires.
    cluster_method  "gmm" (default) or "hdbscan" -- which algorithm n_components
                  fits on the 2-D PCA coordinates. Ignored when the
                  fallback_1d histogram fires (that path is always GMM auto).
    n_components  clustering mode, colouring by cluster (CIFs symlinked into
                  results/tm_reannotated):
                    - cluster_method="gmm": int fits exactly that many
                      components (manual); 'auto' sweeps auto_k_min..auto_k_max
                      and picks the knee of the BIC-vs-k curve (kneed,
                      polynomial smoothing -- see _fit_gmm_bic_sweep), with a
                      BIC diagnostic plot alongside the embedding. Ellipses
                      are drawn from the GMM covariances.
                    - cluster_method="hdbscan": 'auto' searches
                      (hdbscan_min_cluster_size, hdbscan_min_samples,
                      hdbscan_cluster_selection_method, hdbscan_metric) with
                      Optuna/TPE scored by DBCV (hdbscan_n_trials trials --
                      see _fit_hdbscan_dbcv_search), with a DBCV diagnostic
                      plot alongside the embedding; 'manual' fits HDBSCAN
                      directly with the explicit hdbscan_* arguments below
                      (hdbscan_min_cluster_size is then required). Points
                      HDBSCAN calls noise (-1) are shown unclustered.
    max_per_cluster  cap on how many CIFs per cluster get symlinked for
                  reannotation (randomly subsampled, including HDBSCAN's
                  noise cluster). Only used when n_components is set, or
                  when the fallback_1d histogram fires.
    auto_k_min, auto_k_max  BIC sweep range for cluster_method="gmm",
                  n_components='auto' -- also used by the fallback_1d
                  histogram's GMM-auto clustering.
    hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_cluster_selection_method,
    hdbscan_metric  explicit HDBSCAN hyperparameters for
                  cluster_method="hdbscan", n_components='manual'.
    hdbscan_n_trials  Optuna trial budget for cluster_method="hdbscan",
                  n_components='auto'.
    pc_x, pc_y    which PCs to plot (1-indexed). Irrelevant when the
                  fallback_1d histogram fires (PC1 only, by construction).
    fallback_1d   when True (default) and PC1 alone already explains >95%
                  of the variance, PC2 carries almost no real signal, so
                  the usual 2-D scatter is misleading -- plot_pca switches
                  to a 1-D histogram of PC1 split by GMM-auto (BIC-knee)
                  clusters instead (see _plot_pca_1d_fallback), ignoring
                  color_by/cluster_method/n_components/pc_x/pc_y entirely.
                  Set False to always force the normal 2-D plot.
    """
    X, meta = _load_protein(protein)

    enabled = ENABLED_MODELS if models is None else models
    keep = meta["model"].isin([m for m, use in enabled.items() if use]).to_numpy()
    if not keep.any():
        raise ValueError(
            f"models={enabled!r} disables every backend present in this ensemble "
            f"({sorted(meta['model'].unique())}) -- enable at least one")
    X, meta = X[keep], meta.loc[keep].reset_index(drop=True)

    n_pc    = min(max(pc_x, pc_y, 5), X.shape[1])
    pca     = PCA(n_components=n_pc)
    coords  = pca.fit_transform(X)
    evr     = pca.explained_variance_ratio_

    if fallback_1d and evr[0] > 0.95:
        _plot_pca_1d_fallback(
            protein, meta, coords[:, 0], evr[0], method_tag="pca",
            max_per_cluster=max_per_cluster, auto_k_min=auto_k_min, auto_k_max=auto_k_max,
        )
        return

    xy      = coords[:, [pc_x - 1, pc_y - 1]]

    _plot_embedding(
        protein, meta, xy, x_col="pc_x", y_col="pc_y", method_tag="pca",
        method_title=f"PCA  PC{pc_x} vs PC{pc_y}  TM-Ca ({X.shape[1] // 3} atoms)",
        color_by=color_by, cluster_method=cluster_method, n_clusters=n_components,
        max_per_cluster=max_per_cluster, auto_k_min=auto_k_min, auto_k_max=auto_k_max,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size, hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
        hdbscan_metric=hdbscan_metric, hdbscan_n_trials=hdbscan_n_trials,
        marker_size=marker_size, opacity=opacity,
        axis_titles=(f"PC{pc_x}  ({evr[pc_x - 1]:.1%} var)",
                     f"PC{pc_y}  ({evr[pc_y - 1]:.1%} var)"),
    )


def _tm_resids(protein):
    """TM-helix residue numbers (UniProt numbering), in the same order as
    the TM-only Ca columns of aligned_ca_tm.npy / _load_protein's X --
    read from apo's resids.parquet (topology is sequence-based, identical for
    both forms, and apoform always exists)."""
    resids = pd.read_parquet(ALIGN_ROOT / f"{protein}__apo" / "resids.parquet")
    return resids.loc[resids["in_tm"], "resid"].to_numpy()


def _write_ca_trace_pdb(path, coords, resids, chain="A"):
    """Minimal single-model Ca-only trace PDB -- the mean/reference
    structure a porcupine plot's arrows are anchored to. ChimeraX renders
    a cartoon from a Ca-only trace directly (same as any low-res/cryo-EM
    Ca backbone model), so no other atoms are needed here. Residue name is
    a filler ('ALA') -- resids.parquet has residue NUMBERS only, not amino
    acid identity."""
    lines = []
    for i, (xyz, resid) in enumerate(zip(coords, resids), start=1):
        x, y, z = (float(v) for v in xyz)
        lines.append(
            f"ATOM  {i:5d}  CA  ALA {chain}{int(resid):4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    Path(path).write_text("".join(lines))


def _write_porcupine_bild(path, mean_xyz, vectors, color="gold", radius=0.15):
    """ChimeraX BILD arrows, one per TM residue, from its mean position
    along `vectors[i]` (already scaled by the caller) -- BILD is
    ChimeraX's own simple primitive-graphics format (open directly like a
    structure file), the standard way to render a porcupine plot."""
    lines = [f".color {color}\n"]
    for start, vec in zip(mean_xyz, vectors):
        end = start + vec
        lines.append(
            f".arrow {start[0]:.3f} {start[1]:.3f} {start[2]:.3f} "
            f"{end[0]:.3f} {end[1]:.3f} {end[2]:.3f} {radius}\n"
        )
    Path(path).write_text("".join(lines))


# bgColor / cartoon color / 2D-label text color for export_pca_porcupine's
# .cxc -- "dark" is the default (easier on the eyes for long sessions);
# arrow color (gold) is left as its own separate `color` argument since it
# reads fine against either background.
PORCUPINE_THEMES = {
    "dark":  {"bg": "black", "cartoon": "light gray", "label": "white"},
    "light": {"bg": "white", "cartoon": "gray",        "label": "black"},
}


def export_pca_porcupine(protein, pc=1, target_max_length=8.0, scale=None,
                          color="gold", theme="dark", open_chimerax=True):
    """Project one PC of the pooled (apo+holo) TM-Ca PCA -- the same PCA
    plot_pca(protein) fits (pooled apo+holo, TM-Ca only); pc=1 here means
    PC1, matching plot_pca's default pc_x=1 -- onto the ensemble's mean TM
    structure as a ChimeraX porcupine plot: one arrow per TM residue,
    pointing along that residue's contribution to PC`pc`, scaled so the
    longest arrow is `target_max_length` A (or exactly `scale` if given).
    `theme` is a PORCUPINE_THEMES key ("dark" default, or "light") controlling
    the .cxc's background/cartoon/label colors.

    Writes results/figures/<protein>/pca_pc{pc}_mean.pdb (Ca-trace
    reference structure), _porcupine.bild (the arrows) and
    _porcupine.cxc (opens both together, styled like
    NPF_pocket_pipeline/scripts/visualize_tm_angle_chimerax.py's .cxc
    convention) -- and, if open_chimerax (default True, macOS-only, and
    only if the exact ChimeraX-1.11.1.app path below still exists on this
    machine), launches ChimeraX on the .cxc directly, same one-call
    convenience as the plots' own click-to-reveal.
    """
    X, meta = _load_protein(protein)
    n_pc = min(max(pc, 5), X.shape[1])
    pca = PCA(n_components=n_pc)
    pca.fit(X)
    component = pca.components_[pc - 1].reshape(-1, 3)
    mean_xyz  = X.mean(axis=0).reshape(-1, 3)
    resids    = _tm_resids(protein)
    if not (len(resids) == len(mean_xyz) == len(component)):
        raise ValueError(
            f"TM residue count mismatch: resids.parquet has {len(resids)}, "
            f"ensemble has {len(mean_xyz)} Ca -- topology changed since "
            f"tm_helix_alignment.py ran?")

    if scale is None:
        max_norm = np.linalg.norm(component, axis=1).max()
        scale = target_max_length / max_norm if max_norm > 0 else 1.0
    vectors = component * scale

    out_dir = FIG_ROOT / protein
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path  = out_dir / f"pca_pc{pc}_mean.pdb"
    bild_path = out_dir / f"pca_pc{pc}_porcupine.bild"
    cxc_path  = out_dir / f"pca_pc{pc}_porcupine.cxc"

    _write_ca_trace_pdb(pdb_path, mean_xyz, resids)
    _write_porcupine_bild(bild_path, mean_xyz, vectors, color=color)
    colors = PORCUPINE_THEMES[theme]

    evr = pca.explained_variance_ratio_[pc - 1]
    cxc_path.write_text(
        f"# PCA PC{pc} porcupine plot -- {protein}\n"
        f"# {evr:.1%} of variance, {len(resids)} TM residues, "
        f"scale={scale:.2f} (max arrow {target_max_length} A)\n"
        f"# arrows = per-residue contribution to PC{pc}, anchored on the "
        f"ensemble's mean TM structure (Ca-trace only)\n"
        "\n"
        f"open {pdb_path.resolve()}\n"
        f"open {bild_path.resolve()}\n"
        "\n"
        f"set bgColor {colors['bg']}\n"
        "cartoon\n"
        f"color {colors['cartoon']}\n"
        "lighting soft\n"
        f'2dlabel text "{protein}  PC{pc} ({evr:.1%} var)" '
        f"xpos 0.02 ypos 0.96 size 18 color {colors['label']}\n"
        "view\n"
    )

    print(f"[porcupine] {protein} PC{pc} ({evr:.1%} var), {len(resids)} TM residues, "
          f"scale={scale:.3f} -> max arrow {target_max_length} A")
    print(f"[porcupine] wrote {pdb_path.name}, {bild_path.name}, {cxc_path.name} in {out_dir}")

    chimerax_app = Path("/Applications/ChimeraX-1.11.1.app")  # bump the version here if you upgrade ChimeraX
    if open_chimerax and chimerax_app.exists():
        subprocess.run(["open", "-a", str(chimerax_app), str(cxc_path)])
    elif open_chimerax:
        print(f"[porcupine] {chimerax_app} not found -- open {cxc_path} in ChimeraX manually")

    return pdb_path, bild_path, cxc_path




print("Setup done.  plot_pca(protein)  "
      "(protein = BASE name, apo+holo pooled, each already pooling all 6 ABCfold backends)"
      "  (default color_by='model' [alphafold3/boltz/chai1/openfold3/protenix/rosettafold3], "
      "or 'status' [apo/holo] / 'ptm' / 'iptm' / 'seed' / 'rmsd_tm'; cluster_method='gmm'; "
      "pass n_components=k [manual] / n_components='auto' [BIC-knee sweep] for GMM clustering, or "
      "cluster_method='hdbscan' with n_components='auto' [Optuna/DBCV search] / 'manual' "
      "for HDBSCAN clustering; models=ENABLED_MODELS ablation switch picks which of the 6 "
      "backends get pooled)  |  export_pca_porcupine(protein, pc=1) opens a "
      "ChimeraX porcupine plot of that PC on the mean TM structure")
