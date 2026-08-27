"""
mmgbsa/src/write_prod_mdp.py
============================
Tiny helper the Stage 3 job script calls once per (complex, replica) to write
that replica's prod.mdp with its own ld_seed. Kept separate so the SLURM job
script stays pure bash + gmx and never needs the repo importable beyond this
one file.

    python write_prod_mdp.py --seed 12345 --out prod.mdp [--nsteps 50000]
"""
from __future__ import annotations

import argparse

import mdp_templates as mdp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsteps", type=int, default=None, help="override production nsteps (smoke test)")
    args = ap.parse_args()
    with open(args.out, "w") as fh:
        fh.write(mdp.prod_mdp(args.seed, args.nsteps))


if __name__ == "__main__":
    main()
