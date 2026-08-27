"""
mmgbsa/src/decomp_parse.py
==========================
Parser for gmx_MMPBSA's per-residue GB decomposition output
(FINAL_DECOMP_MMPBSA.dat / .csv, idecomp=2, dec_verbose=3).

gmx_MMPBSA's decomp file layout is version-dependent, so this parser is
header-driven and tolerant:
  * finds the `DELTAS` section, then its `Total Energy Decomposition`
    sub-block (also `Sidechain` / `Backbone` when present);
  * reads the block's column header to locate the Avg. column for each of
    Internal / van der Waals / Electrostatic / Polar Solvation /
    Non-Polar Solv. / TOTAL (each term is an Avg./Std.Dev./SEM triplet);
  * accepts residue tokens as either `R:A:LEU:5`, `LEU 5`, `L5`, or
    `A/LEU5` and returns (resname, resid) plus the 6 term averages.

Anything that doesn't fit is skipped with a note rather than raising -- run
`python decomp_parse.py <file> --dump` on a real smoke-test output to see
what was and wasn't picked up, and tighten from there (same "confirm against
a real run" iteration redocking/ used).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

TERMS = ["internal", "van der waals", "electrostatic", "polar solvation", "non-polar solv", "total"]
TERM_KEY = {"internal": "internal", "van der waals": "vdw", "electrostatic": "eel",
            "polar solvation": "egb", "non-polar solv": "esurf", "total": "total"}

_RES_PATTERNS = [
    re.compile(r"^[A-Za-z]:(?P<ch>[A-Za-z0-9]):(?P<rn>[A-Z0-9]{1,4}):(?P<ri>-?\d+)$"),  # R:A:LEU:5
    re.compile(r"^(?P<rn>[A-Z]{2,4})\s+(?P<ri>-?\d+)$"),                                  # LEU 5
    re.compile(r"^(?P<rn>[A-Z]{1,4})(?P<ri>-?\d+)$"),                                     # L5 / LEU5
]


@dataclass
class ResidueRow:
    resname: str
    resid: int
    is_ligand: bool
    internal: float
    vdw: float
    eel: float
    egb: float
    esurf: float
    total: float


def _split(line: str) -> list[str]:
    parts = [c.strip() for c in line.split(",")]
    if len(parts) < 3:
        parts = [c.strip() for c in re.split(r"\s{2,}|\t", line.strip())]
    return parts


def _parse_residue_token(tok: str) -> tuple[str, int] | None:
    tok = tok.strip().strip('"')
    for pat in _RES_PATTERNS:
        m = pat.match(tok)
        if m:
            return m.group("rn").upper(), int(m.group("ri"))
    return None


def _term_columns(header_cells: list[str]) -> dict[str, int]:
    """Map term-key -> index of its Avg. column. gmx_MMPBSA writes the term
    name once, spanning an Avg./SD/SEM triplet, so the Avg. column is the
    term-name cell's own index (subsequent 2 cells blank or SD/SEM)."""
    cols: dict[str, int] = {}
    low = [c.lower() for c in header_cells]
    for i, cell in enumerate(low):
        for term in TERMS:
            if term in cell and TERM_KEY[term] not in cols:
                cols[TERM_KEY[term]] = i
    return cols


def parse_decomp(path: Path, ligand_resname: str = "GA1") -> list[ResidueRow]:
    text = Path(path).read_text(errors="replace").splitlines()
    rows: list[ResidueRow] = []
    in_deltas = False
    in_total_block = False
    term_cols: dict[str, int] = {}

    for line in text:
        low = line.lower()
        if low.strip().startswith("deltas"):
            in_deltas = True
            in_total_block = False
            continue
        if not in_deltas:
            continue
        if "total energy decomposition" in low:
            in_total_block = True
            term_cols = {}
            continue
        if "sidechain energy decomposition" in low or "backbone energy decomposition" in low:
            in_total_block = False
            continue
        if not in_total_block:
            continue

        cells = _split(line)
        if not cells or not cells[0]:
            continue
        if not term_cols:
            tc = _term_columns(cells)
            if {"vdw", "eel", "egb", "esurf", "total"}.issubset(tc):
                term_cols = tc
            continue  # this was the header row

        parsed = _parse_residue_token(cells[0])
        if parsed is None:
            continue
        resname, resid = parsed

        def val(key: str) -> float:
            idx = term_cols.get(key)
            if idx is None or idx >= len(cells):
                return float("nan")
            try:
                return float(cells[idx])
            except ValueError:
                # Avg. might be one cell to the right of the term-name cell
                for j in (idx + 1, idx + 2):
                    if j < len(cells):
                        try:
                            return float(cells[j])
                        except ValueError:
                            pass
                return float("nan")

        rows.append(ResidueRow(
            resname=resname, resid=resid,
            is_ligand=(resname == ligand_resname.upper()),
            internal=val("internal"), vdw=val("vdw"), eel=val("eel"),
            egb=val("egb"), esurf=val("esurf"), total=val("total"),
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--ligand-resname", default="GA1")
    args = ap.parse_args()
    rows = parse_decomp(args.path, args.ligand_resname)
    print(f"parsed {len(rows)} residue rows ({sum(r.is_ligand for r in rows)} ligand)")
    if args.dump:
        for r in rows:
            print(f"  {r.resname:>4} {r.resid:>5}  vdw={r.vdw:+7.3f} eel={r.eel:+8.3f} "
                  f"egb={r.egb:+8.3f} esurf={r.esurf:+7.3f} TOTAL={r.total:+8.3f}"
                  + ("  [LIG]" if r.is_ligand else ""))


if __name__ == "__main__":
    main()
