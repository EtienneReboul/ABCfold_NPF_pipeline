#!/usr/bin/env python3
"""
scripts/inject_mmseqs_msa.py
=============================
Stage 3b of the ABCfold NPF pipeline: copy the MSA/template fields that
scripts/fetch_mmseqs2_msa.py fetched for a protein's APOFORM onto the
matching HOLOFORM fold_input.json, instead of querying the ColabFold
MMseqs2 webserver a second time for the exact same sequence.

Apoform and holoform share one protein chain (built by
scripts/make_af3_input.py with the same --chain-id, default "A") — only the
holoform JSON additionally carries a ligand entity. So this just merges
every extra key mmseqs2msa added to the apoform's protein entry
(unpairedMsa, templates, and anything else that CLI writes) onto the
holoform's protein entry, leaving the holoform's own name/modelSeeds/ligand
entity untouched.

Usage (called by Snakemake rule `inject_mmseqs_msa_holo`):
    python scripts/inject_mmseqs_msa.py \\
        --apo-resolved  data/fold_inputs/NPF6.3_Q05085__apo/fold_input.resolved.json \\
        --holo-base     data/fold_inputs/NPF6.3_Q05085__holo/fold_input.json \\
        --output        data/fold_inputs/NPF6.3_Q05085__holo/fold_input.resolved.json
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--apo-resolved", required=True)
    p.add_argument("--holo-base",    required=True)
    p.add_argument("--output",       required=True)
    return p.parse_args()


def protein_entry(doc: dict) -> dict:
    for entry in doc["sequences"]:
        if "protein" in entry:
            return entry["protein"]
    raise ValueError("No 'protein' entry found in fold_input.json sequences")


def main():
    args = parse_args()

    apo_doc = json.loads(Path(args.apo_resolved).read_text())
    holo_doc = json.loads(Path(args.holo_base).read_text())

    apo_protein = protein_entry(apo_doc)
    holo_protein = protein_entry(holo_doc)

    carried = {k: v for k, v in apo_protein.items() if k not in ("id", "sequence")}
    if not carried:
        raise ValueError(
            f"{args.apo_resolved} has no MSA/template fields beyond id/sequence — "
            "did scripts/fetch_mmseqs2_msa.py actually run mmseqs2msa on it?"
        )
    holo_protein.update(carried)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(holo_doc, indent=2) + "\n")

    print(
        f"[inject_mmseqs_msa] {output_path.parent.name}: copied "
        f"{sorted(carried.keys())} from {Path(args.apo_resolved).parent.name} → {output_path}"
    )


if __name__ == "__main__":
    main()
