#!/usr/bin/env python3
"""Bootstrap/append per-protein markdown+code cell pairs to one of the
4 ligand-category notebooks under notebook/ --
tm_conformation_clustering_{gibberellin,nitrate,other_ligand,apoform}.ipynb
-- for every base protein under results/tm_alignment/ in that category
that doesn't have one yet.

These 4 notebooks replaced the single combined
tm_conformation_clustering.ipynb: running every protein's interactive
Plotly output in one notebook was overloading the notebook renderer
(149MB and growing), mirroring the same split AF3_NPF_pipeline already
uses (tm_conformation_clustering_{gibberellin,nitrate,other_ligand,
apoform}.ipynb there). Each category notebook is otherwise identical --
same setup cells (loading / embedding / clustering / reannotation code),
same one-markdown-cell-per-protein structure -- just a different
`PROTEINS` filter and title.

Run this again any time new proteins appear under results/tm_alignment/;
it only appends cells for proteins not already covered in that category's
notebook, so it's safe to re-run repeatedly (e.g. after each partial sync
from IFB) without touching existing cells. Pass a category positional arg,
or "all" to refresh every category notebook in one go.

The setup cells' shared code lives in scripts/_notebook_setup_functions.py
(the big loading/embedding/clustering cell, unchanged across categories)
and CELL3_TEMPLATE/CELL4_SRC below (ligand metadata + backend summary,
templated only where the category filter applies) -- single source of
truth for all 4 notebooks; edit there and re-run with --refresh-setup to
rewrite an existing notebook's setup cells (0-5) without touching its
per-protein cells.
"""
import argparse
import json
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebook"
ALIGN_ROOT = ROOT / "results" / "tm_alignment"
SETUP_FUNCTIONS_PATH = ROOT / "scripts" / "_notebook_setup_functions.py"

BACKENDS = ["alphafold3", "boltz", "chai1", "openfold3", "protenix", "rosettafold3"]

CATEGORIES = ["gibberellin", "nitrate", "other_ligand", "apoform"]

CATEGORY_DESCRIPTION = {
    "gibberellin": "Gibberellin (GA1) importers",
    "nitrate": "Nitrate transporters",
    "other_ligand": "Other-ligand transporters (ABA / auxin / glycerate / dimethylarsenate / JA-Ile / dipeptide / flavonoid / polyamine)",
    "apoform": "Apoform-only (no ligand assigned)",
}

DISPLAY_NAMES = {
    "GA1": "Gibberellin A1",
    "nitrate": "NO3- (nitrate)",
    "ABA": "(S)-abscisic acid",
    "auxin": "Indole-3-acetic acid (IAA)",
    "glycerate": "D-glycerate",
    "dimethylarsenate": "dimethylarsinate",
    "JA-Ile": "(3R,7S)-jasmonoyl-L-isoleucine",
    "glycylglycine": "Gly-Gly",
    "spermidine": "spermidine",
    "quercetin-3-O-sophoroside": "quercetin-3-O-sophoroside",
}

DEFAULT_KERNELSPEC = {
    "display_name": "npf-notebook",
    "language": "python",
    "name": "python3",
}
DEFAULT_LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.11.15",
}

# Ligand-metadata cell (formerly notebook cell 3): identical across
# categories except the PROTEINS filter and its docstring/print text.
CELL3_TEMPLATE = '''import yaml

_config = yaml.safe_load((ROOT / "config.yaml").read_text())
LIGANDS = _config["ligands"]  # ligand key -> {"smiles": ...}, from config.yaml

# Mirrors worflows/preprocessing/Snakefile's protein->ligand assignment lists.
HC_IMPORTERS = ['NPF3.1', 'NPF4.1', 'NPF2.12', 'NPF2.13', 'NPF2.10', 'NPF2.5']
NITRATE_TRANSPORTERS = ['NPF1.1', 'NPF1.2', 'NPF1.3', 'NPF2.3', 'NPF2.4', 'NPF2.7', 'NPF2.9', 'NPF2.11', 'NPF4.6', 'NPF5.5', 'NPF5.8', 'NPF5.9', 'NPF5.10', 'NPF5.11', 'NPF5.12', 'NPF5.14', 'NPF5.16', 'NPF6.2', 'NPF6.3', 'NPF7.2', 'NPF7.3', 'NPF8.5']
ABA_TRANSPORTERS = ['NPF2.14', 'NPF4.2', 'NPF4.5', 'NPF4.7', 'NPF5.1', 'NPF5.2', 'NPF5.3', 'NPF5.7']
AUXIN_TRANSPORTERS = ['NPF7.1']
GLYCERATE_TRANSPORTERS = ['NPF8.4']
DIMETHYLARSENATE_TRANSPORTERS = ['NPF8.1', 'NPF8.2']
JA_ILE_TRANSPORTERS = ['NPF2.6']
DIPEPTIDE_TRANSPORTERS = ['NPF8.3']
FLAVONOID_TRANSPORTERS = ['NPF2.8']
POLYAMINE_TRANSPORTERS = ['NPF6.4']
LOW_CONFIDENCE_GA_IMPORTERS = ['NPF2.1', 'NPF5.6']


def ligand_for(npf_name):
    """Same precedence as worflows/preprocessing/Snakefile's ligand_for();
    npf_name is the fasta basename, e.g. 'NPF2.12' (not the full
    '<npf_name>_<uniprot>' protein directory name)."""
    if npf_name in HC_IMPORTERS:
        return "GA1"
    if npf_name in NITRATE_TRANSPORTERS:
        return "nitrate"
    if npf_name in ABA_TRANSPORTERS:
        return "ABA"
    if npf_name in AUXIN_TRANSPORTERS:
        return "auxin"
    if npf_name in GLYCERATE_TRANSPORTERS:
        return "glycerate"
    if npf_name in DIMETHYLARSENATE_TRANSPORTERS:
        return "dimethylarsenate"
    if npf_name in DIPEPTIDE_TRANSPORTERS:
        return "glycylglycine"
    if npf_name in FLAVONOID_TRANSPORTERS:
        return "quercetin-3-O-sophoroside"
    if npf_name in POLYAMINE_TRANSPORTERS:
        return "spermidine"
    if npf_name in JA_ILE_TRANSPORTERS:
        return "JA-Ile"
    if npf_name in LOW_CONFIDENCE_GA_IMPORTERS:
        return "GA1"
    return None


def category_of(npf_name):
    """Buckets ligand_for()'s result for this protein's markdown header;
    this notebook only covers the "__CATEGORY__" bucket -- see the
    PROTEINS filter below."""
    key = ligand_for(npf_name)
    if key == "GA1":
        return "gibberellin"
    if key == "nitrate":
        return "nitrate"
    if key is None:
        return "apoform"
    return "other_ligand"


def _base_protein_name(dirname):
    """Strip the '__apo'/'__holo' ABCfold-run suffix, if present, to
    recover the base protein name plot_pca expects --
    they pool both runs internally via _load_protein."""
    if dirname.endswith("__apo") or dirname.endswith("__holo"):
        return dirname.rsplit("__", 1)[0]
    return dirname


ALL_PROTEINS = sorted({
    _base_protein_name(p.name) for p in ALIGN_ROOT.iterdir()
    if p.is_dir() and not p.name.startswith(".")
})

# BASE protein (e.g. "NPF2.12_Q9LFX9") -> ligand key or None (apoform only)
PROTEIN_LIGAND = {
    protein: ligand_for(protein.rsplit("_", 1)[0]) for protein in ALL_PROTEINS
}

# This notebook's slice: only proteins in the "__CATEGORY__" ligand
# category (see category_of() above) -- one of 4 category notebooks split
# out of the old combined tm_conformation_clustering.ipynb.
PROTEINS = [p for p in ALL_PROTEINS if category_of(p.rsplit("_", 1)[0]) == "__CATEGORY__"]

print(f"{len(PROTEINS)} protein(s) under results/tm_alignment/ in the '__CATEGORY__' category")
for protein in PROTEINS:
    key = PROTEIN_LIGAND[protein]
    if key is None:
        print(f"  {protein:20s} apoform only  [{category_of(protein.rsplit('_', 1)[0])}]")
    else:
        print(f"  {protein:20s} holoform ligand: {key}  ({LIGANDS[key]['smiles']})  [{category_of(protein.rsplit('_', 1)[0])}]")
'''

CELL4_SRC = '''print(f"Backends discovered ({list(BACKEND_PATTERNS)}), per-protein frame counts from results/tm_alignment/:")
for protein in PROTEINS:
    counts = {}
    for status in ("apo", "holo"):
        meta_path = ALIGN_ROOT / f"{protein}__{status}" / "meta.parquet"
        if not meta_path.exists():
            continue
        for model, n in pd.read_parquet(meta_path)["model"].value_counts().items():
            counts[model] = counts.get(model, 0) + int(n)
    if counts:
        summary = ", ".join(f"{m}({counts[m]})" for m in BACKEND_PATTERNS if m in counts)
        print(f"  {protein:20s} {summary}")
    else:
        print(f"  {protein:20s} NO DATA")
'''


def _title_md(category, n_proteins):
    desc = CATEGORY_DESCRIPTION[category]
    return f'''# TM-Ca ensemble -- PCA + GMM/HDBSCAN clustering, pooled across all 6 ABCfold backends -- {desc}

**Kernel:** `abcfold-npf-notebook` (`envs/notebook.yaml`)

One of 4 notebooks split out of the former single `tm_conformation_clustering.ipynb`
by ligand category (gibberellin / nitrate / other_ligand / apoform),
mirroring `AF3_NPF_pipeline/notebook/tm_conformation_clustering_{{gibberellin,nitrate,other_ligand,apoform}}.ipynb`
-- running every protein's interactive Plotly output in one notebook was
overloading the notebook renderer, same as in that repo. Same setup cell
(loading / embedding / clustering / reannotation code, unchanged) and
same one-markdown-cell-per-protein structure; this file only covers
**{desc}** ({n_proteins} protein(s) currently under `results/tm_alignment/`
-- see `scripts/generate_notebook_protein_cells.py {category}` to refresh
as more land).

`scripts/tm_helix_alignment.py` already pools every backend's CIFs (across
every seed x diffusion/sample) into one `results/tm_alignment/<protein>__{{apo,holo}}/`
ensemble per run, tagging each frame with a `model` column
(`alphafold3`/`boltz`/`chai1`/`openfold3`/`protenix`/`rosettafold3`) -- so
unlike `AF3_NPF_pipeline`'s notebooks (AF3 only, `color_by="status"` was
the only interesting axis, and a separate `_boltz` fork was needed just to
overlay a second model), **`color_by="model"` is the key new axis here**
and every `plot_*` function below defaults to it: a single backend's
diffusion doesn't always recover every conformation a second one finds
(that's the whole reason this pipeline exists instead of extending
`AF3_NPF_pipeline`).

```python
plot_pca(protein)                                 # colour by model (default) -- which of the 6 backends produced each point
plot_pca(protein, color_by='status')              # colour by status instead (apo/holo)
plot_pca(protein, color_by='rmsd_tm')             # colour by per-frame RMSD (A) to the converged TM-helix mean

plot_pca(protein, n_components=3)                                 # GMM manual: fit GMM-3
plot_pca(protein, n_components='auto')                            # GMM auto: BIC-knee sweep k=1..20

plot_pca(protein, cluster_method='hdbscan', n_components='auto')  # HDBSCAN auto: Optuna/DBCV search
plot_pca(protein, cluster_method='hdbscan', n_components='manual', # HDBSCAN manual: explicit params
         hdbscan_min_cluster_size=15)

# Ablation: which backends get pooled before the PCA fit -- e.g. is AF3 still
# needed once OpenFold3 is in the mix? Toggle ENABLED_MODELS (defined in the
# setup cell) or pass a one-off dict via `models=` without touching the global.
plot_pca(protein, models={{**ENABLED_MODELS, "alphafold3": False}})
```

**A caveat worth watching for, visible via `color_by='model'`:** backends
don't all contribute the same number of frames per protein -- e.g.
OpenFold3 currently produces far more samples per seed than the other 5
backends for at least one protein in this ensemble, which can dominate a
joint PCA fit or GMM/HDBSCAN clustering pass by sheer point count. Worth
keeping in mind when reading a cluster's backend composition, not
something this notebook corrects for.

Another one: RosettaFold3 writes both a `..._model.cif` and a
`..._model_fixed.cif` per (seed, sample) -- two distinct CIFs with
near-identical coordinates that `scripts/tm_helix_alignment.py`'s
`parse_frame_id()` doesn't distinguish (its regex only captures
seed/sample), so both get pooled as separate frames sharing one
`frame_id` -- effectively near-duplicating RosettaFold3's weight in this
ensemble. `_reannotate` below tolerates the resulting symlink-name
collision (skips the second one rather than erroring) but doesn't
deduplicate the underlying frames.

Also worth noting: `find_confidence()` in `scripts/tm_helix_alignment.py`
now does a real per-backend pTM/ipTM lookup (each of the 6 backends writes
its confidence JSON/npz under a different name/layout -- see that function
for the per-backend mapping), so `color_by='ptm'` and `color_by='iptm'` both
work. One asymmetry to expect on apoform (single-chain) runs: AlphaFold3
reports iptm as null (no interface to score) -- correctly NaN here -- while
the other 5 backends report 0.0 for the same case instead.

**Prerequisite:** run `worflows/postprocessing/Snakefile` (stage 6,
`scripts/tm_helix_alignment.py`) first for every protein you want to look
at here -- this notebook only reads `results/tm_alignment/`, it does not
compute alignments or touch `results/abcfold/` directly (except to
rediscover CIFs for reannotation symlinks, see `_reannotate` in the setup
cell).
'''


def _metadata_section_md(category):
    return f'''## Ligand / protein metadata

`ligand_for()` mirrors `worflows/preprocessing/Snakefile`'s function of the
same name (that file is a Snakefile -- uses `checkpoint`/`rule`/
`configfile` directives -- not a plain importable module, so the
protein->ligand lists are mirrored here rather than imported; keep the two
in sync if `ligand_for()` changes). `category_of()` buckets its result into
`"gibberellin"` (GA1), `"nitrate"`, `"other_ligand"` (every other assigned
ligand) or `"apoform"` (no ligand assigned at all) -- `PROTEINS` below is
filtered to just `"{category}"`, unlike the old combined
`tm_conformation_clustering.ipynb` (see the title cell above). Ligand
SMILES are read directly from `config.yaml`'s `ligands:` section.
'''


def _per_protein_section_md(category):
    return f'''## Per-protein cells

One markdown + code cell pair per `"{category}"`-category protein
discovered under `results/tm_alignment/` at the time this notebook was
built. Each code cell: a plain `color_by='model'` PCA first (which of the
6 backends produced each point, no clustering), then an HDBSCAN/DBCV-tuned
clustering pass, then an AF3-excluded ablation. Edit a cell's arguments
directly (`color_by`, `cluster_method`, manual
`n_clusters`/`hdbscan_min_cluster_size`, etc.) for one-off exploration on
that protein without touching any other cell.

As more `"{category}"`-category proteins' stage 6
(`scripts/tm_helix_alignment.py`) output lands under `results/tm_alignment/`,
re-run `scripts/generate_notebook_protein_cells.py {category}` to append
their markdown+code cell pairs here (safe to re-run repeatedly -- it only
appends cells for proteins not already covered).
'''


def _new_cell_id():
    return secrets.token_hex(4)


def _md_cell(text):
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "id": _new_cell_id(), "metadata": {}, "source": lines}


def _code_cell(text, outputs=None):
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _new_cell_id(),
        "metadata": {},
        "outputs": outputs or [],
        "source": lines,
    }


def _setup_cells(category, n_proteins):
    cell3_src = CELL3_TEMPLATE.replace("__CATEGORY__", category)
    return [
        _md_cell(_title_md(category, n_proteins)),
        _code_cell(SETUP_FUNCTIONS_PATH.read_text()),
        _md_cell(_metadata_section_md(category)),
        _code_cell(cell3_src),
        _code_cell(CELL4_SRC),
        _md_cell(_per_protein_section_md(category)),
    ]


def _new_notebook(category, n_proteins):
    return {
        "cells": _setup_cells(category, n_proteins),
        "metadata": {"kernelspec": DEFAULT_KERNELSPEC, "language_info": DEFAULT_LANGUAGE_INFO},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _notebook_path(category):
    return NOTEBOOK_DIR / f"tm_conformation_clustering_{category}.ipynb"


def _load_ligand_metadata(category):
    """Exec this category's CELL3_TEMPLATE in an isolated namespace to get
    ligand_for()/category_of()/PROTEINS/LIGANDS without duplicating those
    lists a third time here."""
    ns = {"ROOT": ROOT, "ALIGN_ROOT": ALIGN_ROOT}
    exec(CELL3_TEMPLATE.replace("__CATEGORY__", category), ns)
    return ns


def _base_protein_name(dirname):
    if dirname.endswith("__apo") or dirname.endswith("__holo"):
        return dirname.rsplit("__", 1)[0]
    return dirname


def _covered_proteins(nb):
    covered = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        m = re.match(r"### `([^`]+)`", src)
        if m:
            covered.add(m.group(1))
    return covered


def _backends_present(protein):
    import pandas as pd
    present = set()
    for status in ("apo", "holo"):
        meta_path = ALIGN_ROOT / f"{protein}__{status}" / "meta.parquet"
        if not meta_path.exists():
            continue
        present |= set(pd.read_parquet(meta_path)["model"].dropna().unique())
    return present


def _build_cell_pair(protein, npf_name, ns):
    ligand_key = ns["ligand_for"](npf_name)
    apo_dir = ALIGN_ROOT / f"{protein}__apo"
    holo_dir = ALIGN_ROOT / f"{protein}__holo"
    holo_exists = holo_dir.exists()

    present_backends = _backends_present(protein)
    missing_backends = [b for b in BACKENDS if b not in present_backends]

    md_lines = []
    if ligand_key is not None:
        display = DISPLAY_NAMES.get(ligand_key, ligand_key)
        avail = "apo + holo data available" if holo_exists else "apoform-only data currently available"
        md_lines.append(f"### `{protein}` -- holoform ligand: **{ligand_key}** ({display}) -- {avail}\n")
        md_lines.append("\n")
        smiles = ns["LIGANDS"][ligand_key]["smiles"]
        md_lines.append(f"`{smiles}`\n")
    else:
        avail = "apo + holo data available" if holo_exists else "apoform only (no ligand assigned)"
        md_lines.append(f"### `{protein}` -- {avail}\n")

    md_lines.append("\n")
    run_paths = f"`results/tm_alignment/{protein}__apo`"
    if holo_exists:
        run_paths += f", `results/tm_alignment/{protein}__holo`"
    md_lines.append(f"{run_paths}.\n")

    if not holo_exists and ligand_key is not None:
        md_lines.append(
            "Holoform hasn't been run yet, so `_load_protein` below only has the\n"
            "apoform ensemble to pool -- `color_by='status'` would show a single\n"
            "category; `color_by='model'` (the default) is where the signal is right\n"
            "now.\n"
        )

    if missing_backends:
        md_lines.append(
            f"\n**Missing backend(s) as of this cell's generation: {', '.join(missing_backends)}** -- "
            "`_load_protein` pools whichever backends have output; `color_by='model'` "
            "will simply show no points for a missing backend rather than erroring. "
            "Re-run `scripts/generate_notebook_protein_cells.py` (or just re-run this "
            "cell) once the gap is backfilled.\n"
        )

    md_cell = {"cell_type": "markdown", "id": _new_cell_id(), "metadata": {}, "source": md_lines}
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": _new_cell_id(),
        "metadata": {},
        "outputs": [],
        "source": [
            f'plot_pca("{protein}")  # no clustering, colored by model (default) -- which of the 6 backends produced each point\n',
            f'plot_pca("{protein}", cluster_method="gmm", n_components="auto")\n',
            f'plot_pca("{protein}", models={{**ENABLED_MODELS, "alphafold3": False}})  # ablation: AF3 excluded -- does the remaining ensemble still cover its conformations?',
        ],
    }
    return md_cell, code_cell


def generate(category, dry_run=False, refresh_setup=False):
    ns = _load_ligand_metadata(category)
    category_proteins = ns["PROTEINS"]

    notebook_path = _notebook_path(category)
    if notebook_path.exists():
        nb = json.loads(notebook_path.read_text())
        if refresh_setup:
            per_protein_cells = nb["cells"][6:]
            nb["cells"] = _setup_cells(category, len(category_proteins)) + per_protein_cells
    else:
        nb = _new_notebook(category, len(category_proteins))

    covered = _covered_proteins(nb)
    new_proteins = [p for p in category_proteins if p not in covered]

    print(f"[{category}] {len(category_proteins)} protein(s) in category, {len(covered)} already covered")
    if not new_proteins and not refresh_setup:
        print(f"[{category}] Nothing to do.")
        return

    for protein in new_proteins:
        if not (ALIGN_ROOT / f"{protein}__apo").exists():
            print(f"[{category}]   SKIP {protein}: no __apo run under results/tm_alignment/ (expected for every protein)")
            continue
        npf_name = protein.rsplit("_", 1)[0]
        md_cell, code_cell = _build_cell_pair(protein, npf_name, ns)
        print(f"[{category}]   + {protein}")
        if not dry_run:
            nb["cells"].append(md_cell)
            nb["cells"].append(code_cell)

    if dry_run:
        print(f"[{category}] (dry run -- notebook not modified)")
        return

    notebook_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"[{category}] Wrote {notebook_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("category", choices=CATEGORIES + ["all"], help="Which category notebook to update")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be added, don't write the notebook")
    parser.add_argument("--refresh-setup", action="store_true", help="Also rewrite the setup cells (0-5) from the current templates, keeping per-protein cells")
    args = parser.parse_args()

    categories = CATEGORIES if args.category == "all" else [args.category]
    for category in categories:
        generate(category, dry_run=args.dry_run, refresh_setup=args.refresh_setup)


if __name__ == "__main__":
    main()
