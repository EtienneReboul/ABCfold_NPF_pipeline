#!/usr/bin/env python3
"""
scripts/make_af3_input.py
==========================
Stage 2 of the ABCfold NPF pipeline:
Generate ONE base fold_input.json per protein x apo/holo form, covering
every replica (seed) at once. This is AlphaFold3-dialect JSON — unchanged
from AF3_NPF_pipeline — because ABCfold accepts exactly this format as its
own input (https://github.com/rigdenlab/ABCFold): the same "name",
"sequences", "modelSeeds", "dialect", "version" JSON drives AlphaFold3,
Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3 together.

This file has no MSA or templates embedded yet — scripts/fetch_mmseqs2_msa.py
(stage 3) adds those from the ColabFold MMseqs2 webserver, once per base
protein, producing fold_input.resolved.json.

Usage (called by Snakemake rule `prepare_af3_input`):
    python scripts/make_af3_input.py \\
        --fasta            data/sequences/NPF6.3_Q05085.fasta \\
        --protein-name      NPF6.3_Q05085 \\
        --output            data/fold_inputs/NPF6.3_Q05085/fold_input.json \\
        --n-replicas        5 \\
        --seed-strategy      sequential \\
        --seed-base          1 \\
        --dialect            alphafold3 \\
        --json-version       1
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from Bio import SeqIO  # pyright: ignore[reportMissingImports]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",              required=True)
    p.add_argument("--protein-name",       required=True)
    p.add_argument("--output",             required=True)
    p.add_argument("--n-replicas",         type=int, required=True,
                    help="j — number of independent seeds (modelSeeds entries)")
    p.add_argument("--seed-strategy",      choices=["sequential", "random"],
                    default="sequential")
    p.add_argument("--seed-base",          type=int, default=1,
                    help="sequential: seeds = seed_base .. seed_base+n_replicas-1")
    p.add_argument("--random-master-seed", type=int, default=0,
                    help="random: RNG seed, combined with a hash of --protein-name "
                         "so every protein gets its own draw")
    p.add_argument("--dialect",            default="alphafold3")
    p.add_argument("--json-version",       type=int, default=1)
    p.add_argument("--chain-id",           default="A")
    p.add_argument("--ligand-smiles",      default=None,
                    help="Co-fold this ligand (holoform); omit for apoform")
    p.add_argument("--ligand-name",        default=None,
                    help="Ligand identifier, used for logging only")
    p.add_argument("--ligand-chain-id",    default="B")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_sequence(fasta_path: Path) -> str:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise RuntimeError(f"No sequence found in {fasta_path}")
    return str(records[0].seq)


def generate_seeds(strategy: str, n_replicas: int, seed_base: int,
                    random_master_seed: int, protein_name: str) -> list[int]:
    """Return `n_replicas` distinct, explicit model seeds."""
    if strategy == "sequential":
        return [seed_base + i for i in range(n_replicas)]

    # "random": deterministic per protein, but not a simple arithmetic
    # sequence — combine the master seed with a stable hash of the protein
    # name so re-running preprocessing reproduces the same seeds.
    digest = hashlib.sha256(protein_name.encode()).hexdigest()
    protein_hash = int(digest[:8], 16)
    rng = random.Random(random_master_seed + protein_hash)
    return rng.sample(range(1, 2**31 - 1), n_replicas)


def build_fold_input(protein_name: str, sequence: str, seeds: list[int],
                      chain_id: str, dialect: str, json_version: int,
                      ligand_smiles: str | None = None, # pyright: ignore[reportGeneralTypeIssues]
                      ligand_chain_id: str = "B") -> dict:
    sequences = [
        {
            "protein": {
                "id": [chain_id],
                "sequence": sequence,
            }
        }
    ]
    if ligand_smiles:
        sequences.append(
            {
                "ligand": {
                    "id": [ligand_chain_id],
                    "smiles": ligand_smiles,
                }
            }
        )
    return {
        "name": protein_name,
        "sequences": sequences,
        "modelSeeds": seeds,
        "dialect": dialect,
        "version": json_version,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    sequence = load_sequence(Path(args.fasta))
    seeds = generate_seeds(
        args.seed_strategy, args.n_replicas, args.seed_base,
        args.random_master_seed, args.protein_name,
    )
    doc = build_fold_input(
        args.protein_name, sequence, seeds, args.chain_id,
        args.dialect, args.json_version,
        ligand_smiles=args.ligand_smiles, ligand_chain_id=args.ligand_chain_id,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2) + "\n")

    form = f"holoform ({args.ligand_name or 'ligand'})" if args.ligand_smiles else "apoform"
    print(
        f"[af3_input] {args.protein_name}: {len(seeds)} seeds "
        f"({args.seed_strategy}, first={seeds[0]}, last={seeds[-1]}), {form} → {output_path}"
    )


if __name__ == "__main__":
    main()
