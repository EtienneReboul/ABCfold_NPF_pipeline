#!/usr/bin/env python3
"""
scripts/fetch_mmseqs2_msa.py
=============================
Stage 3 of the ABCfold NPF pipeline — the "default run" MSA/template
resolution: MSA + top-hit templates from the ColabFold MMseqs2 webserver,
no manual curation, no pocket restraint. This is the ABCfold-native
equivalent of NPF_pocket_pipeline's `templates.default_run`
(worflows/preprocessing/Snakefile there uses scripts/run_msa.py +
scripts/make_default_boltz_input.py to the same end for Boltz-2) — here it
wraps ABCfold's own `mmseqs2msa` CLI utility instead of re-implementing the
ColabFold ticket/poll protocol, since ABCfold already ships exactly that
tool for its own AlphaFold3-dialect JSON.

Run ONCE per base protein, on the apoform fold_input.json (holoform shares
the same sequence — scripts/inject_mmseqs_msa.py copies the resulting
unpairedMsa/templates fields across instead of re-querying the webserver).
Resolving this locally (with internet) rather than passing ABCfold's own
`--mmseqs2` flag at processing time means the SLURM compute nodes
(worflows/processing/submit_abcfold.sh) never need outbound network access.

Usage (called by Snakemake rule `fetch_mmseqs2_msa`):
    python scripts/fetch_mmseqs2_msa.py \\
        --input-json   data/fold_inputs/NPF6.3_Q05085__apo/fold_input.json \\
        --output-json  data/fold_inputs/NPF6.3_Q05085__apo/fold_input.resolved.json \\
        --num-templates 20 \\
        --retries 3 \\
        --delay 8
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json",    required=True)
    p.add_argument("--output-json",   required=True)
    p.add_argument("--num-templates", type=int, default=20)
    p.add_argument("--retries",       type=int, default=3)
    p.add_argument("--delay",         type=float, default=8,
                   help="Seconds to sleep after a successful call (politeness "
                        "towards the shared ColabFold webserver)")
    return p.parse_args()


def main():
    args = parse_args()

    mmseqs2msa = shutil.which("mmseqs2msa")
    if mmseqs2msa is None:
        raise RuntimeError(
            "mmseqs2msa not found on PATH — install the `abcfold` package "
            "(see envs/preprocessing.yaml) to get this CLI entry point."
        )

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs2msa,
        "--input_json", str(input_json),
        "--output_json", str(output_json),
        "--templates",
        "--num_templates", str(args.num_templates),
    ]

    last_err = None
    for attempt in range(1, args.retries + 1):
        print(f"[mmseqs2_msa] {input_json.parent.name}: attempt {attempt}/{args.retries} "
              f"— {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            if not output_json.exists():
                raise RuntimeError(f"mmseqs2msa exited 0 but {output_json} was not written")
            print(f"[mmseqs2_msa] done → {output_json}")
            time.sleep(args.delay)
            return
        except (subprocess.CalledProcessError, RuntimeError) as e:
            last_err = e
            if attempt < args.retries:
                wait = 30 * attempt
                print(f"[mmseqs2_msa] attempt {attempt} failed ({e}); "
                      f"waiting {wait}s before retry ...", flush=True)
                time.sleep(wait)

    raise RuntimeError(
        f"mmseqs2msa failed after {args.retries} attempts for {input_json}: {last_err}"
    ) from last_err


if __name__ == "__main__":
    sys.exit(main())
