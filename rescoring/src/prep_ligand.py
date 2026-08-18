#!/usr/bin/env python3
"""
rescoring/src/prep_ligand.py
===============================
Per-ligand parameterization: generates params/<ligand_key>.params (Rosetta
ligand params, via rdkit_to_params) for every ligand key actually present in
data/manifest.csv, and validates the positional-atom-order correction
(ligand_fix.py) against a handful of OTHER complexes for that ligand,
deliberately spread across as many different ABCfold backends as possible.

Generalized from the sibling project's single-ligand, "run once by hand and
eyeball it" version: this pipeline co-folds 11 different ligands across 6
backends, so parameterization has to run per ligand, and the thing most at
risk of not generalizing (backend-consistent positional atom ordering — see
ligand_fix.py's module docstring) needs to actually be exercised across
backends in validation, not just across proteins on one backend. Made an
ordinary (idempotent, re-runnable) step rather than a manual one, consistent
with how the rest of this pipeline is automated -- still worth reading the
printed validation report once per ligand the first time it runs.

Usage:
    python prep_ligand.py [--n-validate 12]
    python prep_ligand.py --ligand GA1   # just one ligand
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

import config
import ligand_fix as lf
import pose_prep as pp

BACKEND_PATTERNS = {
    "alphafold3":   "alphafold3",
    "boltz":        "boltz",
    "chai1":        "chai",
    "openfold3":    "openfold",
    "protenix":     "protenix",
    "rosettafold3": "rosettafold",
}


def _backend_of(complex_id: str) -> str:
    for backend, pattern in BACKEND_PATTERNS.items():
        if pattern in complex_id:
            return backend
    return "unknown"


def _spread_across_backends(rows: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample up to n rows, spread across as many distinct backends as
    possible -- validation must actually exercise the multi-backend
    positional-atom-order assumption ligand_fix.py relies on, not just
    multiple proteins on a single backend."""
    rows = rows.assign(_backend=rows["complex_id"].map(_backend_of))
    backends = sorted(rows["_backend"].unique())
    per_backend = max(1, n // max(1, len(backends)))
    picked_frames = [
        g.sample(n=min(per_backend, len(g)), random_state=seed)
        for _, g in rows.groupby("_backend")
    ]
    picked = pd.concat(picked_frames) if picked_frames else rows.iloc[0:0]
    if len(picked) < n:
        remaining = rows.drop(picked.index)
        n_extra = min(n - len(picked), len(remaining))
        if n_extra:
            picked = pd.concat([picked, remaining.sample(n=n_extra, random_state=seed)])
    return picked.drop(columns="_backend")


def _renumber_add_ring_indices(params_path) -> None:
    """rdkit_to_params numbers every ring RDKit perceives (including plain
    aromatic rings, which never get their own ADD_RING line) when assigning
    each flexible (non-aromatic, pucker-sampled) ring's index -- so a
    ligand with aromatic rings interleaved among its flexible ones (e.g.
    quercetin-3-O-sophoroside: 2 aromatic rings + 3 pucker-sampled ones)
    ends up with ADD_RING indices like 2, 4, 5 instead of a contiguous
    1, 2, 3. Rosetta's own ring-conformer-database loader indexes into its
    lookup table by that literal number and errors ("Cannot load database
    file: An invalid ring size was provided") on the resulting gaps —
    confirmed by hand: renumbering quercetin-3-O-sophoroside.params's rings
    2/4/5 -> 1/2/3 with no other change made it load cleanly. Renumbering
    every ADD_RING line's index contiguously (order of appearance -> 1..N)
    fixes this for any ligand it affects; a no-op for ligands whose rings
    were already contiguous (nothing to renumber, or none at all)."""
    text = Path(params_path).read_text()
    lines = text.splitlines()
    new_lines = []
    next_index = 1
    for line in lines:
        if line.startswith("ADD_RING "):
            parts = line.split(maxsplit=2)
            line = f"ADD_RING {next_index} {parts[2]}"
            next_index += 1
        new_lines.append(line)
    Path(params_path).write_text("\n".join(new_lines) + "\n")


def generate_params(mol: Chem.Mol, ligand_key: str, resname: str, out_path=None) -> Path:
    """Writes `out_path` (default: config.params_path(ligand_key), the
    canonical location) -- callers doing a multi-attempt search (see
    _build_params_with_self_check) pass a per-attempt temp path instead, so
    each attempt's output doesn't clobber the previous one before the best
    is chosen."""
    import pyrosetta
    pyrosetta.init("-mute all")
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem
    from rdkit_to_params import Params

    mol = _Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    atomnames = {i: a.GetPDBResidueInfo().GetName() for i, a in enumerate(mol.GetAtoms())}
    p = Params.from_mol(mol, name=resname, atomnames=atomnames)
    out_path = Path(out_path) if out_path is not None else config.params_path(ligand_key)
    p.dump(str(out_path))
    _renumber_add_ring_indices(out_path)
    return out_path


MAX_PARAMS_ATTEMPTS = 8
_PROBE_SCRIPT = Path(__file__).resolve().parent / "_prep_ligand_probe_subprocess.py"


def _run_attempt(ligand_key: str, seed: int, ligand_chain: str, cif_paths: list[str],
                  out_params_path: Path) -> dict:
    """Run ONE idealized-conformer attempt in its own subprocess (see
    _prep_ligand_probe_subprocess.py's module docstring for why a
    subprocess is required, not just a nicety): builds params/<ligand_key>
    at `out_params_path`, then stages+loads+scores every cif_paths[i]
    against it. Returns the subprocess's JSON result dict:
    {n_atoms, formula, results: [[complex_id, ok, score_or_error], ...]}."""
    args_path = config.STAGED_DIR / f"_prep_args_{ligand_key}_{seed}.json"
    result_path = config.STAGED_DIR / f"_prep_result_{ligand_key}_{seed}.json"
    args_path.write_text(json.dumps({
        "ligand_key": ligand_key, "seed": seed, "ligand_chain": ligand_chain,
        "out_params_path": str(out_params_path), "cif_paths": cif_paths,
    }))
    proc = subprocess.run(
        [sys.executable, str(_PROBE_SCRIPT), str(args_path), str(result_path)],
        cwd=_PROBE_SCRIPT.parent, capture_output=True, text=True,
    )
    args_path.unlink(missing_ok=True)
    if not result_path.exists():
        # Subprocess crashed before writing a result at all (e.g. params generation
        # itself raised) -- treat as every complex failing, so the retry loop moves on.
        return {"n_atoms": None, "formula": None,
                "results": [[Path(p).stem, False, proc.stderr.strip()[-2000:]] for p in cif_paths]}
    result = json.loads(result_path.read_text())
    result_path.unlink(missing_ok=True)
    return result


def validate_one_ligand(ligand_key: str, rows: pd.DataFrame, n_validate: int) -> None:
    """Build params/<ligand_key>.params and validate it against up to
    n_validate real complexes (excluding a held-out reference row from
    scope, spread across backends -- see _spread_across_backends), all in
    ONE sample -- deliberately a single sample used for both the
    accept/reject decision AND the reported validation, not two separate
    draws (an earlier version probed a small sample then validated a
    different, larger one -- confirmed by hand that a params attempt could
    pass a 5-complex probe and still fail a *different* 6-complex sample
    right after, since idealized-conformer generation isn't perfectly
    reproducible even with a fixed seed; using one sample throughout
    removes that inconsistency by construction).

    Each attempt (fresh idealized conformer, fresh subprocess -- see
    _run_attempt) is scored by how many of the sample complexes it loads
    successfully. Stops early on a perfect attempt; otherwise keeps
    retrying up to MAX_PARAMS_ATTEMPTS and keeps the best (fewest failures)
    attempt seen. Accepts (copies the best attempt's params to the
    canonical params/<ligand_key>.params) if its failure rate is <=50% --
    matching real-pose noise this pipeline already expects and reports
    per-complex (see README.md's raw-pose caveat), not evidence the
    correction method itself is broken. Exits fatally only if even the best
    attempt fails on more than half the sample, or if ligand_fix.py's own
    positional-correction check ever raises (a real correctness bug, not
    geometry noise -- see build_corrected_ligand_mol's docstring)."""
    smiles = config.load_ligand_smiles(ligand_key)
    resname = config.ligand_resname(ligand_key)
    template = lf.build_template(smiles, resname)
    reference_row = rows.iloc[0]
    ligand_chain, _smiles2 = pp.resolved_ligand_chain(config.ABCFOLD_OUT_ROOT / f"{reference_row['protein']}__holo")

    other_rows = rows[rows["complex_id"] != reference_row["complex_id"]]
    sample_rows = _spread_across_backends(other_rows, n_validate)
    backends_hit = sorted(sample_rows["complex_id"].map(_backend_of).unique())
    cif_by_stem = {Path(p).stem: (cid, p) for cid, p in zip(sample_rows["complex_id"], sample_rows["cif_path"])}
    cif_paths = [str(config.PIPELINE_ROOT / p) for p in sample_rows["cif_path"]]
    print(f"[prep_ligand] {ligand_key} (resname={resname}): validating against {len(sample_rows)} "
          f"complex(es) spanning backend(s) {backends_hit}...")

    best = None  # (n_failures, attempt, result_dict)
    for attempt in range(MAX_PARAMS_ATTEMPTS):
        tmp_params = config.PARAMS_DIR / f"_attempt_{ligand_key}_{attempt}.params"
        result = _run_attempt(ligand_key, 42 + attempt, ligand_chain, cif_paths, tmp_params)
        mismatches = [(cif_by_stem[stem][0], msg) for stem, ok, msg in result["results"]
                      if not ok and isinstance(msg, str) and msg.startswith("mismatch:")]
        if mismatches:
            # ligand_fix.py's own positional-correction check failed on a REAL pose --
            # a real correctness bug (see its module docstring), not idealized-geometry
            # noise a different seed could fix. No point retrying; fail loudly now.
            tmp_params.unlink(missing_ok=True)
            sys.exit(f"[prep_ligand] {ligand_key}: VALIDATION FAILED (positional-correction mismatch) for "
                      f"{len(mismatches)}/{len(sample_rows)} complex(es) -- this indicates ligand_fix.py's "
                      "cross-backend positional atom-order assumption doesn't hold for this ligand:\n"
                      + "\n".join(f"  {cid}: {msg}" for cid, msg in mismatches))
        n_fail = sum(1 for _, ok, _ in result["results"] if not ok)
        if best is None or n_fail < best[0]:
            if best is not None:
                best[2].get("_tmp_params", Path("/dev/null")).unlink(missing_ok=True)
            result["_tmp_params"] = tmp_params
            best = (n_fail, attempt, result)
        else:
            tmp_params.unlink(missing_ok=True)
        if n_fail == 0:
            break
        print(f"[prep_ligand] {ligand_key}: attempt {attempt + 1} (seed={42 + attempt}) -- "
              f"{n_fail}/{len(sample_rows)} complex(es) failed to load/score"
              + ("" if attempt < MAX_PARAMS_ATTEMPTS - 1 else " (out of attempts)"))

    n_fail, best_attempt, result = best
    if n_fail > len(sample_rows) / 2:
        best[2]["_tmp_params"].unlink(missing_ok=True)
        failures = [(cif_by_stem[stem][0], msg) for stem, ok, msg in result["results"] if not ok]
        sys.exit(f"[prep_ligand] {ligand_key}: VALIDATION FAILED -- best of {MAX_PARAMS_ATTEMPTS} idealized-"
                  f"conformer attempts still failed {n_fail}/{len(sample_rows)} complexes (>50%, systematic, "
                  "not isolated pose noise):\n" + "\n".join(f"  {cid}: {msg}" for cid, msg in failures))

    final_params = config.params_path(ligand_key)
    result["_tmp_params"].replace(final_params)
    if n_fail:
        print(f"[prep_ligand] {ligand_key}: accepted attempt {best_attempt + 1} with {n_fail}/"
              f"{len(sample_rows)} complex(es) still failing (kept as isolated real-structure noise -- "
              "these specific complexes will also fail at rescoring time, logged there the same way):")
        for stem, ok, msg in result["results"]:
            if not ok:
                print(f"[prep_ligand]     {cif_by_stem[stem][0]}: {msg}")
    elif best_attempt:
        print(f"[prep_ligand] {ligand_key}: accepted attempt {best_attempt + 1} (all "
              f"{len(sample_rows)} complexes OK)")

    expected_n_atoms, expected_formula = result["n_atoms"], result["formula"]
    print(f"[prep_ligand] {ligand_key}: {template.GetNumAtoms()} heavy atoms, "
          f"{expected_n_atoms} total after AddHs, formula {expected_formula}")
    for stem, ok, val in result["results"]:
        if ok:
            complex_id, _ = cif_by_stem[stem]
            print(f"[prep_ligand]   validated {complex_id} ({_backend_of(complex_id)}): "
                  f"OK (total_score={val:.1f})")

    naming = {
        "ligand_key": ligand_key,
        "resname": resname,
        "smiles": smiles,
        "reference_complex_id": reference_row["complex_id"],
        "n_heavy": template.GetNumAtoms(),
        "n_total": expected_n_atoms,
        "formula": expected_formula,
    }
    config.atom_naming_path(ligand_key).write_text(json.dumps(naming, indent=2))
    print(f"[prep_ligand] {ligand_key}: {len(sample_rows) - n_fail}/{len(sample_rows)} validated OK -- "
          f"wrote {final_params}, {config.atom_naming_path(ligand_key)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-validate", type=int, default=12,
                     help="number of additional (non-reference) complexes to spot-check, per ligand")
    ap.add_argument("--ligand", action="append", help="restrict to this ligand key (repeatable)")
    args = ap.parse_args()

    if not config.MANIFEST_CSV.exists():
        sys.exit(f"{config.MANIFEST_CSV} not found -- run make_manifest.py first.")
    manifest = pd.read_csv(config.MANIFEST_CSV)

    ligand_keys = args.ligand if args.ligand else sorted(manifest["ligand"].unique())
    for ligand_key in ligand_keys:
        rows = manifest[manifest["ligand"] == ligand_key]
        if rows.empty:
            print(f"[prep_ligand] {ligand_key}: no complexes in manifest, skipping")
            continue
        validate_one_ligand(ligand_key, rows, args.n_validate)

    print("[prep_ligand] done.")


if __name__ == "__main__":
    main()
