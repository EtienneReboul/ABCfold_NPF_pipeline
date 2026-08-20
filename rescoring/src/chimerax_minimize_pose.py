"""
ChimeraX Script: staged-PDB minimization & PDB export
========================================================
Generalized from NPF_pocket_pipeline/scripts/minimize_cif.py -- runs inside
ChimeraX itself (session/run() are only available in a live ChimeraX
process), so this file cannot import anything from rescoring/src/ (a
different Python environment). Its input is expected to be a staged PDB
already produced by sanitize_for_chimerax.py: protein chain A + a
bond-order-correct ligand (chain L) with CONECT records, so ChimeraX's
IDATM atom-typing reads real bonds instead of re-perceiving them from
raw 3-D proximity.

Usage (called by run_chimerax_try.py, but also works standalone):

    chimerax --nogui --offscreen --script "chimerax_minimize_pose.py /path/to/staged.pdb /path/to/minimized.pdb"

What it does:
    1. Opens the staged PDB
    2. Runs energy minimization (skipped if the output PDB already exists)
    3. Logs the energy trajectory to <output_stem>_energy.csv
    4. Saves the minimized structure as PDB
"""

import sys
import csv
import re
from pathlib import Path

# ChimeraX imports (available at runtime, but not recognized by static analysis)
from chimerax.core.commands import run  # pyright: ignore[reportMissingImports]
from chimerax.core.logger import StringPlainTextLog  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Energy log helpers
# ---------------------------------------------------------------------------

def parse_energy_log(log_text: str) -> list[dict]:
    """
    Parse energy entries from the minimize logEnergy output.

    ChimeraX logs lines like:
        'Step 100, energy: -12345.67 kJ/mol'

    Returns a list of dicts: [{step: int, energy_kJ_mol: float}, ...]
    """
    entries = []

    for step, energy in re.findall(
        r"[Ss]tep\s+(\d+)[,:\s]+energy[:\s=]+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        log_text,
    ):
        entries.append({"step": int(step), "energy_kJ_mol": float(energy)})

    if entries:
        return entries

    for step, energy in re.findall(
        r"(\d+)\D+([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)",
        log_text,
    ):
        entries.append({"step": int(step), "energy_kJ_mol": float(energy)})

    return entries


def save_energy_csv(entries: list[dict], path: Path, session) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "energy_kJ_mol"])
        writer.writeheader()
        writer.writerows(entries)
    session.logger.info(
        f"  Energy trajectory saved ({len(entries)} steps): {path.name}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(session):
    if len(sys.argv) < 3:
        session.logger.error(
            "Usage: chimerax --nogui --script chimerax_minimize_pose.py <input.pdb> <output.pdb>"
        )
        raise SystemExit(1)

    in_path = Path(sys.argv[1]).expanduser().resolve()
    pdb_path = Path(sys.argv[2]).expanduser().resolve()

    if not in_path.exists():
        session.logger.error(f"Input structure not found: {in_path}")
        raise SystemExit(1)

    if pdb_path.exists():
        session.logger.info(
            f"Output PDB already exists -- skipping minimization: {pdb_path.name}"
        )
        run(session, "quit")
        return

    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    session.logger.info(f"Opening: {in_path.name}")
    run(session, f"open {str(in_path)!r}")

    session.logger.info("Running minimization...")
    with StringPlainTextLog(session.logger) as log:
        run(session, "minimize #1 liveUpdates false logEnergy true")
        minimize_log = log.getvalue()

    session.logger.info(f"[minimize log]:\n{minimize_log.strip()}")

    energy_entries = parse_energy_log(minimize_log)
    if energy_entries:
        energy_csv = pdb_path.with_name(pdb_path.stem + "_energy.csv")
        save_energy_csv(energy_entries, energy_csv, session)
    else:
        session.logger.warning(
            "Could not parse energy values from minimize log.\n"
            f"Raw log:\n{minimize_log.strip()}"
        )

    session.logger.info(f"Saving minimized PDB: {pdb_path.name}")
    run(session, f"save {str(pdb_path)!r} models #1 format pdb")

    run(session, "close #1")
    run(session, "quit")


# ChimeraX injects `session` at module scope at runtime
main(session)  # pyright: ignore[reportUndefinedVariable] # noqa: F821
