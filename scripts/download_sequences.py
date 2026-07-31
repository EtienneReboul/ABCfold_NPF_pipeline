#!/usr/bin/env python3
"""
scripts/download_sequences.py
==============================
Stage 1 of the ABCfold NPF pipeline:
  1. Download all Arabidopsis NPF sequences from UniProt (Swiss-Prot reviewed).
  2. Split into per-protein FASTA files.

Unchanged from AF3_NPF_pipeline: this does NOT submit anything to a webserver
itself — scripts/fetch_mmseqs2_msa.py (stage 3) is what queries the ColabFold
MMseqs2 API, once per base protein, and only for proteins this stage
discovers.

Called by Snakemake rule `download_sequences` (worflows/preprocessing/Snakefile).
Writes a sentinel file listing all discovered proteins.

Usage:
    python scripts/download_sequences.py \\
        --fasta-dir data/sequences \\
        --sentinel  data/sequences/sequences.done \\
        --query     'reviewed:true AND organism_id:3702 AND protein_name:"NRT1/ PTR FAMILY"' \\
        --size      500
"""

import argparse
import re
from pathlib import Path

import requests  # pyright: ignore[reportMissingModuleSource]
from Bio import SeqIO  # pyright: ignore[reportMissingImports]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta-dir", required=True)
    p.add_argument("--sentinel",  required=True,
                   help="File to write listing all protein base-names on success")
    p.add_argument("--query",     required=True)
    p.add_argument("--size",      type=int, default=500)
    return p.parse_args()


# ── UniProt ────────────────────────────────────────────────────────────────────

def download_uniprot_fasta(query: str, size: int, out_fasta: Path) -> None:
    if out_fasta.exists():
        print(f"[sequences] UniProt FASTA exists — skipping download: {out_fasta}")
        return

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "format": "fasta", "size": size}
    chunks = []

    print("[sequences] Downloading NPF sequences from UniProt ...")
    while url:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"UniProt request failed ({r.status_code}): {r.text}")
        chunk = r.text.strip()
        if chunk:
            chunks.append(chunk)
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split("<")[1].split(">")[0]
            params = {}
        else:
            url = None

    if not chunks:
        raise RuntimeError("No sequences returned — check UniProt query syntax.")

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.write_text("\n".join(chunks) + "\n")
    n = sum(1 for line in "\n".join(chunks).splitlines() if line.startswith(">"))
    print(f"[sequences] Downloaded {n} sequences → {out_fasta}")


# ── Sequence helpers ───────────────────────────────────────────────────────────

def parse_uniprot_id(record_id: str) -> str:
    parts = record_id.split("|")
    return parts[1] if len(parts) >= 2 else record_id


def parse_gene_name(description: str) -> str | None: # pyright: ignore[reportGeneralTypeIssues]
    m = re.search(r"GN=(\S+)", description)
    return m.group(1) if m else None


def base_name(record) -> str:
    uid = parse_uniprot_id(record.id)
    gene = parse_gene_name(record.description)
    return f"{gene}_{uid}" if gene else uid


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    fasta_dir = Path(args.fasta_dir)
    fasta_dir.mkdir(parents=True, exist_ok=True)

    all_fasta = fasta_dir / "npf_arabidopsis.fasta"
    download_uniprot_fasta(args.query, args.size, all_fasta)

    records = list(SeqIO.parse(all_fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences in {all_fasta}")
    print(f"[sequences] Loaded {len(records)} sequences.")

    names = []
    for record in records:
        name = base_name(record)
        names.append(name)
        per_fasta = fasta_dir / f"{name}.fasta"
        if not per_fasta.exists():
            uid = parse_uniprot_id(record.id)
            gene = parse_gene_name(record.description)
            header = f">{uid} GN={gene}" if gene else f">{uid}"
            per_fasta.write_text(f"{header}\n{record.seq}\n")

    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("\n".join(names) + "\n")
    print(f"[sequences] Done. {len(names)} proteins. Sentinel written: {sentinel}")


if __name__ == "__main__":
    main()
