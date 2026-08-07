#!/usr/bin/env python3
"""Append per-protein markdown+code cell pairs to
notebook/tm_conformation_clustering.ipynb for every base protein under
results/tm_alignment/ that doesn't have one yet.

tm_conformation_clustering.ipynb's own "Per-protein cells" section
(cell 5) is explicitly hand-maintained -- one markdown+code cell pair per
protein, added as each protein's stage 6 (scripts/tm_helix_alignment.py)
output lands -- unlike per_model_pca_clustering.ipynb /
model_capability_exploration.ipynb, which loop over the fixed 6-backend
set and pool proteins internally. This script automates that manual
step: run it again any time new proteins appear under
results/tm_alignment/; it only appends cells for proteins not already
covered, so it's safe to re-run repeatedly (e.g. after each partial sync
from IFB) without touching existing hand-edited cells.

Ligand metadata (ligand_for/category_of/LIGANDS) is executed straight out
of the notebook's own "Ligand / protein metadata" code cell rather than
duplicated a third time here (notebook cell 3 already mirrors
worflows/preprocessing/Snakefile's ligand_for(), which says to "keep the
two in sync" -- adding a generator-script copy would be a third place to
drift).

Each generated code cell's third line calls plot_pca with an AF3-excluded
`models=` override, so every new protein comes with a ready-made ablation
comparison (does OpenFold3 + the rest already cover AF3's conformations?)
without having to hand-add it later.
"""
import argparse
import json
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebook" / "tm_conformation_clustering.ipynb"
ALIGN_ROOT = ROOT / "results" / "tm_alignment"

BACKENDS = ["alphafold3", "boltz", "chai1", "openfold3", "protenix", "rosettafold3"]

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


def _load_ligand_metadata_cell(nb):
    """Exec notebook cell 3 ("Ligand / protein metadata") in an isolated
    namespace to get ligand_for()/LIGANDS without duplicating those lists
    here. Needs ROOT and yaml only."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith("import yaml"):
            ns = {"ROOT": ROOT, "ALIGN_ROOT": ALIGN_ROOT}
            exec("".join(cell["source"]), ns)
            return ns
    raise RuntimeError("Could not find the 'Ligand / protein metadata' code cell in the notebook")


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
    present = set()
    for status in ("apo", "holo"):
        meta_path = ALIGN_ROOT / f"{protein}__{status}" / "meta.parquet"
        if not meta_path.exists():
            continue
        import pandas as pd
        present |= set(pd.read_parquet(meta_path)["model"].dropna().unique())
    return present


def _new_cell_id():
    return secrets.token_hex(4)


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

    md_cell = {
        "cell_type": "markdown",
        "id": _new_cell_id(),
        "metadata": {},
        "source": md_lines,
    }
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be added, don't write the notebook")
    args = parser.parse_args()

    nb = json.loads(NOTEBOOK_PATH.read_text())
    ns = _load_ligand_metadata_cell(nb)

    all_proteins = sorted({
        _base_protein_name(p.name) for p in ALIGN_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    })
    covered = _covered_proteins(nb)
    new_proteins = [p for p in all_proteins if p not in covered]

    if not new_proteins:
        print(f"Nothing to do -- all {len(all_proteins)} protein(s) under results/tm_alignment/ already have a cell pair.")
        return

    print(f"{len(new_proteins)} new protein(s) to add ({len(covered)} already covered):")
    for protein in new_proteins:
        if not (ALIGN_ROOT / f"{protein}__apo").exists():
            print(f"  SKIP {protein}: no __apo run under results/tm_alignment/ (expected for every protein)")
            continue
        npf_name = protein.rsplit("_", 1)[0]
        md_cell, code_cell = _build_cell_pair(protein, npf_name, ns)
        print(f"  + {protein}")
        if not args.dry_run:
            nb["cells"].append(md_cell)
            nb["cells"].append(code_cell)

    if args.dry_run:
        print("\n(dry run -- notebook not modified)")
        return

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"\nWrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
