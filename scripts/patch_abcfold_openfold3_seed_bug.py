#!/usr/bin/env python3
"""
Patch ABCFold's abcfold/openfold3/run_openfold3.py to fix the OpenFold3
seed-duplication bug: run_openfold() writes runner_yaml ONCE (with every
modelSeed baked into experiment_settings.seeds) and then reuses that same
file for every per-seed `run_openfold predict` subprocess call, so each of
the n invocations independently re-predicts all n seeds (n^2 total runs
instead of n).

Fix: scope the runner yaml to a single seed per loop iteration.

Upstream bug report: https://github.com/rigdenlab/ABCFold (issue TBD —
see the draft at scripts/abcfold_openfold3_seed_issue.md).

Usage — run with whichever `python3` resolves the same `abcfold` install
ABCfold itself runs with (on IFB this is the plain `/usr/bin/python3`
resolving `~/.local/lib/python3.12/site-packages/abcfold`; the script
locates the target file via `importlib.util.find_spec`, so just make sure
`python3 -c "import abcfold"` succeeds in whatever shell you run this from):

    python3 patch_abcfold_openfold3_seed_bug.py            # apply
    python3 patch_abcfold_openfold3_seed_bug.py --check     # dry-run, no write
    python3 patch_abcfold_openfold3_seed_bug.py --revert    # restore from .bak
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

OLD = '''    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input yaml file and msa to the output directory")
            working_dir = output_dir

        openfold_json = OpenfoldJson(working_dir)
        openfold_json.json_to_json(input_json)
        runner_yaml = working_dir / "openfold3_runner.yml"
        openfold_json.write_yaml(runner_yaml)

        for seed in openfold_json.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.json")

            openfold_json.write_json(out_file)
            logger.info("Running OpenFold 3 using seed: %s", seed)
            openfold_out_dir = output_dir / f"openfold_results_seed-{seed}"
            cmd = (
                generate_openfold_command(
                    out_file,
                    openfold_out_dir,
                    runner_yaml,
                    openfold_ckpt,
                    number_of_models
                )
                if not test
                else generate_openfold_test_command()
            )'''

NEW = '''    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input yaml file and msa to the output directory")
            working_dir = output_dir

        openfold_json = OpenfoldJson(working_dir)
        openfold_json.json_to_json(input_json)
        all_seeds = list(openfold_json.seeds)

        for seed in all_seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.json")

            openfold_json.write_json(out_file)

            # Scope the runner yaml to THIS seed only. Upstream bug: writing
            # one runner_yaml (with every modelSeed) before this loop and
            # reusing it for every iteration made each per-seed invocation
            # re-predict every seed (n^2 total runs instead of n).
            openfold_json.seeds = [seed]
            runner_yaml = working_dir / f"openfold3_runner_seed-{seed}.yml"
            openfold_json.write_yaml(runner_yaml)

            logger.info("Running OpenFold 3 using seed: %s", seed)
            openfold_out_dir = output_dir / f"openfold_results_seed-{seed}"
            cmd = (
                generate_openfold_command(
                    out_file,
                    openfold_out_dir,
                    runner_yaml,
                    openfold_ckpt,
                    number_of_models
                )
                if not test
                else generate_openfold_test_command()
            )'''


def find_target() -> Path:
    spec = importlib.util.find_spec("abcfold.openfold3.run_openfold3")
    if spec is None or spec.origin is None:
        sys.exit(
            "ERROR: could not import abcfold.openfold3.run_openfold3 — "
            "activate the same environment ABCfold itself runs in first."
        )
    return Path(spec.origin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry-run, don't write")
    ap.add_argument("--revert", action="store_true", help="restore from .bak")
    args = ap.parse_args()

    target = find_target()
    backup = target.with_suffix(target.suffix + ".bak")
    print(f"Target: {target}")

    if args.revert:
        if not backup.exists():
            sys.exit(f"ERROR: no backup found at {backup}")
        shutil.copy2(backup, target)
        print(f"Reverted {target} from {backup}")
        return

    text = target.read_text()

    if NEW in text:
        print("Already patched — nothing to do.")
        return

    if OLD not in text:
        sys.exit(
            "ERROR: expected code block not found verbatim in the installed "
            "file (ABCFold source may have changed upstream since this patch "
            "was written). Inspect run_openfold() in the target file by hand."
        )

    patched = text.replace(OLD, NEW)

    if args.check:
        print("Patch would apply cleanly. Re-run without --check to write.")
        return

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backed up original to {backup}")

    target.write_text(patched)
    print("Patched successfully.")
    print(
        "Verify with: python -c \"import abcfold.openfold3.run_openfold3 as m; "
        "print(open(m.__file__).read())\" | grep -A2 'openfold_json.seeds = \\[seed\\]'"
    )


if __name__ == "__main__":
    main()
