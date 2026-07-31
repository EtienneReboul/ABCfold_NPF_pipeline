#!/usr/bin/env python3
"""
scripts/run_deeptmhmm_topology.py
====================================
Run DeepTMHMM (via BioLib cloud) on all NPF proteins and write
tm_topology_summary.json for use by scripts/tm_helix_alignment.py.

If --gff3 already exists the BioLib call is skipped and the cached
GFF3 is parsed directly (resumable runs, no repeated API charges).

Exits with code 1 and a clear message if any protein does not have
exactly --expected-tm TM helices, halting the Snakemake pipeline
before any downstream alignment runs on bad topology data.

Usage (called by Snakemake rule run_deeptmhmm_topology):
    python scripts/run_deeptmhmm_topology.py \\
        --fasta        data/sequences/npf_arabidopsis.fasta \\
        --gff3         data/interpro/deeptmhmm_TMRs.gff3 \\
        --topology     data/interpro/tm_topology_summary.json \\
        --sentinel     data/interpro/deeptmhmm_topology.done \\
        --expected-tm  12
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO # pyright: ignore[reportMissingImports]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",       required=True)
    p.add_argument("--gff3",        required=True, help="Path to cache/store the DeepTMHMM GFF3")
    p.add_argument("--topology",    required=True, help="Output tm_topology_summary.json")
    p.add_argument("--sentinel",    required=True)
    p.add_argument("--expected-tm", type=int, default=12,
                   help="Required TM helix count per protein (default: 12)")
    return p.parse_args()


# ── FASTA helpers ──────────────────────────────────────────────────────────────

def parse_uniprot_id(record_id: str) -> str:
    parts = record_id.split("|")
    return parts[1] if len(parts) >= 2 else record_id


def parse_gene_name(description: str) -> str | None: # pyright: ignore[reportGeneralTypeIssues]
    m = re.search(r"GN=(\S+)", description)
    return m.group(1) if m else None


def protein_base_name(record) -> str:
    uid  = parse_uniprot_id(record.id)
    gene = parse_gene_name(record.description)
    return f"{gene}_{uid}" if gene else uid


# ── DeepTMHMM via BioLib ───────────────────────────────────────────────────────

def run_deeptmhmm(fasta_path: Path, gff3_out: Path) -> None:
    """
    Call `biolib run DTU/DeepTMHMM` from a temp directory and copy the
    resulting TMRs.gff3 to gff3_out. BioLib writes output to
    biolib_results/ relative to CWD, so we isolate it in a tmpdir.

    BioLib mangles absolute paths on the cloud side, so we copy the FASTA
    into the tmpdir and pass only the filename (relative path).
    """
    print(f"[deeptmhmm] Submitting {fasta_path} to DeepTMHMM via BioLib ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fasta = Path(tmpdir) / fasta_path.name
        shutil.copy(fasta_path, tmp_fasta)
        subprocess.run(
            ["biolib", "run", "DTU/DeepTMHMM", "--fasta", fasta_path.name],
            cwd=tmpdir,
            check=True,
        )
        gff3_src = Path(tmpdir) / "biolib_results" / "TMRs.gff3"
        if not gff3_src.exists():
            raise RuntimeError(
                f"DeepTMHMM finished but TMRs.gff3 not found in {tmpdir}/biolib_results/. "
                "Check BioLib output above for errors."
            )
        gff3_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(gff3_src, gff3_out)
    print(f"[deeptmhmm] GFF3 saved to {gff3_out}")


# ── GFF3 parser ────────────────────────────────────────────────────────────────

def parse_gff3(gff3_path: Path) -> dict[str, list[dict]]:
    """
    Parse DeepTMHMM GFF3 output.
    Returns {uid: [{"start": int, "end": int}, ...]} keyed by UniProt accession.

    DeepTMHMM GFF3 format (tab-separated, no ##sequence-region header):
        sp|Q9M390|PTR1_ARATH  TMhelix  30  53  ...
    The UID is the second pipe-delimited field of the sequence identifier.
    """
    helices: dict[str, list[dict]] = {}
    for line in gff3_path.read_text().splitlines():
        if line.startswith("#") or not line.strip() or line == "//":
            continue
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] != "TMhelix":
            continue
        seq_id = parts[0]
        uid    = seq_id.split("|")[1] if "|" in seq_id else seq_id
        helices.setdefault(uid, []).append(
            {"start": int(parts[2]), "end": int(parts[3])}
        )
    return helices


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    fasta    = Path(args.fasta)
    gff3_out = Path(args.gff3)
    expected = args.expected_tm

    # ── 1. Build uid → protein_name mapping from FASTA ────────────────────────
    records = list(SeqIO.parse(fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences found in {fasta}")
    uid_to_name = {parse_uniprot_id(r.id): protein_base_name(r) for r in records}
    print(f"[deeptmhmm] {len(records)} sequences loaded from {fasta}")

    # ── 2. Run DeepTMHMM (or use cached GFF3) ─────────────────────────────────
    if gff3_out.exists():
        print(f"[deeptmhmm] Using cached GFF3: {gff3_out}")
    else:
        run_deeptmhmm(fasta, gff3_out)

    # ── 3. Parse GFF3 ─────────────────────────────────────────────────────────
    helix_by_uid = parse_gff3(gff3_out)

    # ── 4. Validate TM count and build summary ────────────────────────────────
    tm_summary: dict[str, dict] = {}
    failures:   list[str]       = []

    for rec in records:
        uid  = parse_uniprot_id(rec.id)
        name = uid_to_name[uid]
        helices = helix_by_uid.get(uid, [])
        n = len(helices)

        print(f"[deeptmhmm] {name}: {n} TM helices predicted", end="")
        if n != expected:
            print(f"  <- ERROR (expected {expected})")
            failures.append(f"  {name}: {n} TM predicted (expected {expected})")
        else:
            print()

        tm_summary[name] = {"DEEPTMHMM": sorted(helices, key=lambda h: h["start"])}

    # ── 5. Failsafe: abort if any protein is wrong ────────────────────────────
    if failures:
        print(
            f"\n[deeptmhmm] ERROR: {len(failures)} protein(s) do not have "
            f"{expected} TM helices:\n" + "\n".join(failures) +
            "\n\nPipeline halted. Inspect the GFF3 at:\n  " + str(gff3_out) +
            "\nDelete it to trigger a fresh DeepTMHMM run, or adjust --expected-tm."
        )
        sys.exit(1)

    # ── 6. Write outputs ───────────────────────────────────────────────────────
    summary_path = Path(args.topology)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(tm_summary, indent=2))
    print(f"\n[deeptmhmm] Topology written: {summary_path}")

    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        f"DeepTMHMM topology done. {len(tm_summary)} proteins, "
        f"all with {expected} TM helices.\n"
    )
    print(f"[deeptmhmm] Done. Sentinel: {sentinel}")


if __name__ == "__main__":
    main()
