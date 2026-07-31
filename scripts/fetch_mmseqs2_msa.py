#!/usr/bin/env python3
"""
scripts/fetch_mmseqs2_msa.py
=============================
Stage 3 of the ABCfold NPF pipeline — the "default run" MSA/template
resolution: MSA + top-hit templates from the ColabFold MMseqs2 webserver,
no manual curation, no pocket restraint. This is the ABCfold-native
equivalent of NPF_pocket_pipeline's `templates.default_run`
(worflows/preprocessing/Snakefile there uses scripts/run_msa.py +
scripts/make_default_boltz_input.py to the same end for Boltz-2).

Two ways to resolve a protein, tried in order:

  1. LOCAL REUSE — if --pocket-pipeline-dir points at that sibling repo and
     it already ran the same default_run for this protein, its cached
     results are reused directly instead of re-querying the webserver:
       data/msa/a3m/{base}.a3m               → unpairedMsa
       data/msa/pdb/{base}.m8                → template hit table
       data/templates/default/{base}/*.cif   → cached template structures
     Same webserver, same query sequence, same result — just no network
     round trip or rate limiting. This reimplements the exact filtering/
     cropping/alignment logic ABCfold's own `mmseqs2msa` CLI uses
     (abcfold.scripts.add_mmseqs_msa.get_templates), just sourcing the raw
     MSA/hit-table/structure inputs locally instead of from the webserver.
     Templates beyond what the sibling cached are topped up with a plain
     RCSB download (no MMseqs2 webserver hit) up to --num-templates.

  2. WEBSERVER — ABCfold's own `mmseqs2msa` CLI utility (ships with the
     abcfold PyPI package), for anything local reuse didn't cover.

Run ONCE per base protein, on the apoform fold_input.json (holoform shares
the same sequence — scripts/inject_mmseqs_msa.py copies the resulting
unpairedMsa/templates fields across instead of re-querying the webserver).
Resolving this locally (with internet, or better yet not needing it at all)
rather than passing ABCfold's own `--mmseqs2` flag at processing time means
the SLURM compute nodes (worflows/processing/submit_abcfold.sh) never need
outbound network access.

Usage (called by Snakemake rule `fetch_mmseqs2_msa`):
    python scripts/fetch_mmseqs2_msa.py \\
        --input-json   data/fold_inputs/NPF6.3_Q05085__apo/fold_input.json \\
        --output-json  data/fold_inputs/NPF6.3_Q05085__apo/fold_input.resolved.json \\
        --num-templates 20 \\
        --retries 3 \\
        --delay 8 \\
        --pocket-pipeline-dir ../NPF_pocket_pipeline
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json",    required=True)
    p.add_argument("--output-json",   required=True)
    p.add_argument("--num-templates", type=int, default=20)
    p.add_argument("--retries",       type=int, default=3)
    p.add_argument("--delay",         type=float, default=8,
                   help="Seconds to sleep after a successful webserver call "
                        "(politeness towards the shared ColabFold webserver)")
    p.add_argument("--pocket-pipeline-dir", default="",
                   help="Path to the NPF_pocket_pipeline sibling repo, to reuse "
                        "its already-fetched default-run MSA/templates instead "
                        "of hitting the webserver. Empty disables local reuse.")
    return p.parse_args()


# ── Local reuse (NPF_pocket_pipeline sibling) ───────────────────────────────────

def protein_entry(doc: dict) -> dict:
    for entry in doc["sequences"]:
        if "protein" in entry:
            return entry["protein"]
    raise ValueError("No 'protein' entry found in fold_input.json sequences")


def base_name_of(name: str) -> str:
    for suffix in ("__apo", "__holo"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_templates_from_local(query_seq: str, m8_path: Path, cached_cif_dir: Path,
                                num_templates: int) -> list[dict]:
    from abcfold.scripts.abc_script_utils import (align_and_map,
                                                    extract_sequence_from_mmcif,
                                                    get_mmcif)
    from abcfold.scripts.add_mmseqs_msa import fetch_mmcif

    tested_pdbs: list[str] = []
    templates: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for line in m8_path.read_text().splitlines():
            if len(templates) >= num_templates:
                break
            cols = line.split()
            if len(cols) < 10:
                continue
            pdb_chain, qid, alilen = cols[1], float(cols[2]), float(cols[3])
            tstart, tend = int(cols[8]), int(cols[9])
            coverage = alilen / len(query_seq)
            pdb_id, chain_id = pdb_chain.split("_", 1)

            # Same template filters as ABCfold/AF3: skip near-identical
            # trivial hits, skip near-zero coverage, one template per PDB ID.
            if (qid == 1.0 and coverage >= 0.95) or coverage < 0.1 or pdb_id in tested_pdbs:
                continue

            cached_cif = cached_cif_dir / f"{pdb_id.upper()}.cif"
            try:
                if cached_cif.exists():
                    cif_str = get_mmcif(str(cached_cif), pdb_id, chain_id, tstart, tend, tmpdir)
                    source = "cached"
                else:
                    cif_str = fetch_mmcif(pdb_id, chain_id, tstart, tend, tmpdir)
                    source = "rcsb"
            except Exception as e:
                print(f"[fetch_mmseqs2_msa/local] {pdb_id}_{chain_id}: skip ({e})", flush=True)
                continue

            template_seq = extract_sequence_from_mmcif(StringIO(cif_str))
            query_indices, template_indices = align_and_map(query_seq, template_seq)
            templates.append({
                "mmcif": cif_str,
                "queryIndices": query_indices,
                "templateIndices": template_indices,
            })
            tested_pdbs.append(pdb_id)
            print(f"[fetch_mmseqs2_msa/local] template {pdb_id}_{chain_id} ({source})", flush=True)

    return templates


def try_local_reuse(input_json: Path, output_json: Path, pocket_pipeline_dir: str,
                     num_templates: int) -> bool:
    if not pocket_pipeline_dir:
        return False

    pocket_dir = Path(pocket_pipeline_dir)
    doc = json.loads(input_json.read_text())
    base = base_name_of(doc["name"])

    a3m_path = pocket_dir / "data" / "msa" / "a3m" / f"{base}.a3m"
    m8_path = pocket_dir / "data" / "msa" / "pdb" / f"{base}.m8"
    if not (a3m_path.exists() and m8_path.exists()):
        return False

    protein = protein_entry(doc)
    query_seq = protein["sequence"]
    cached_cif_dir = pocket_dir / "data" / "templates" / "default" / base

    templates = build_templates_from_local(query_seq, m8_path, cached_cif_dir, num_templates)

    protein["unpairedMsa"] = a3m_path.read_text()
    protein["pairedMsa"] = ""
    protein["templates"] = templates

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"[fetch_mmseqs2_msa/local] {base}: reused MSA from {a3m_path}, "
        f"{len(templates)} template(s) → {output_json}",
        flush=True,
    )
    return True


# ── Webserver fallback ───────────────────────────────────────────────────────────

def fetch_from_webserver(input_json: Path, output_json: Path, num_templates: int,
                          retries: int, delay: float) -> None:
    mmseqs2msa = shutil.which("mmseqs2msa")
    if mmseqs2msa is None:
        raise RuntimeError(
            "mmseqs2msa not found on PATH — install the `abcfold` package "
            "(see envs/preprocessing.yaml) to get this CLI entry point."
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs2msa,
        "--input_json", str(input_json),
        "--output_json", str(output_json),
        "--templates",
        "--num_templates", str(num_templates),
    ]

    last_err = None
    for attempt in range(1, retries + 1):
        print(f"[mmseqs2_msa] {input_json.parent.name}: attempt {attempt}/{retries} "
              f"— {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            if not output_json.exists():
                raise RuntimeError(f"mmseqs2msa exited 0 but {output_json} was not written")
            print(f"[mmseqs2_msa] done → {output_json}")
            time.sleep(delay)
            return
        except (subprocess.CalledProcessError, RuntimeError) as e:
            last_err = e
            if attempt < retries:
                wait = 30 * attempt
                print(f"[mmseqs2_msa] attempt {attempt} failed ({e}); "
                      f"waiting {wait}s before retry ...", flush=True)
                time.sleep(wait)

    raise RuntimeError(
        f"mmseqs2msa failed after {retries} attempts for {input_json}: {last_err}"
    ) from last_err


def main():
    args = parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)

    if try_local_reuse(input_json, output_json, args.pocket_pipeline_dir, args.num_templates):
        return

    fetch_from_webserver(input_json, output_json, args.num_templates, args.retries, args.delay)


if __name__ == "__main__":
    sys.exit(main())
