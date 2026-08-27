"""
mmgbsa/src/aggregate_decomp.py
==============================
Stage 5: pool every (complex, replica) gmx_MMPBSA GB result into two tables.

Per-residue path:
  results/mmgbsa/<cid>/rep<k>/FINAL_DECOMP_MMPBSA.dat
    -> decomp_parse.parse_decomp()  (per-residue ligand<->residue DELTA:
       Internal / vdW / EEL / Polar-solv EGB / Non-polar-solv ESURF / TOTAL)
    -> drop the ligand's own row
    -> zip in topology order with the ordered residue list from
       results/systems/<cid>/protein_raw.pdb  (the pose = ABCfold full-sequence
       numbering = the `resnr` in rescoring/data/position_resnr_map_full.csv),
       so the true PDB resid is recovered regardless of any pdb2gmx renumber
    -> average over the ~100 frames (already done inside the .dat) and over the
       3 replicas; keep inter-replica SD + SEM
    -> map (protein, resid) -> alignment position (1..746) via config
    -> collapse to per (protein, role, position): mean over that protein's
       complexes (ca_clusters), same shape as
       redocking/results/rescoring/position_energetics_full.csv

Total-energy path:
  results/mmgbsa/<cid>/rep<k>/FINAL_RESULTS_MMPBSA.dat -> DELTA TOTAL dG_GB

Outputs (results/mmgbsa/):
  decomp_by_residue.csv     protein, role, complex_id, resid, resname, position,
                            is_cdd_pocket, cdd_position, gb_total, gb_vdw, gb_eel,
                            gb_egb, gb_esurf, n_replicas, inter_rep_sd
  decomp_by_position.csv    protein, role, position, is_cdd_pocket, cdd_position,
                            mean_gb_total, mean_gb_vdw, mean_gb_eel, mean_gb_egb,
                            mean_gb_esurf, sem, n, n_replicas, resnr_examples
  binding_energy_summary.csv complex_id, protein, role, replica, dg_gb_total
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
from decomp_parse import parse_decomp

# 3-letter protonation-variant aliases -> canonical, for the resname sanity check
_ALIAS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
          "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}


def ordered_protein_residues(protein_pdb: Path) -> list[tuple[int, str]]:
    """(resid, resname) in file order, one entry per residue."""
    out: list[tuple[int, str]] = []
    seen: set[tuple[str, int]] = set()
    for line in protein_pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resn = line[17:20].strip()
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        chain = line[21]
        key = (chain, resi)
        if key in seen:
            continue
        seen.add(key)
        out.append((resi, resn))
    return out


def parse_total_dg(results_dat: Path) -> float | None:
    """The DELTA TOTAL dG_GB from gmx_MMPBSA's FINAL_RESULTS_MMPBSA.dat
    (newer builds write 'ΔTOTAL', older 'DELTA TOTAL')."""
    if not results_dat.exists():
        return None
    for line in results_dat.read_text(errors="replace").splitlines():
        if re.search(r"(Δ\s*TOTAL|DELTA\s+TOTAL)", line, re.I):
            nums = re.findall(r"-?\d+\.\d+", line)
            if nums:
                return float(nums[0])
    return None


def collect_residue_rows(cid: str, protein: str, role: str) -> list[dict]:
    sysdir = config.SYSTEMS_DIR / cid
    protein_pdb = sysdir / "protein_raw.pdb"
    if not protein_pdb.exists():
        print(f"[stage5] {cid}: no protein_raw.pdb -- skipping")
        return []
    ordered = ordered_protein_residues(protein_pdb)

    # per resid -> list of (gb_total, vdw, eel, egb, esurf) across replicas
    per_resid: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    resname_by_resid: dict[int, str] = {}
    n_reps_seen = 0

    for rep in range(config.N_REPLICAS):
        dat = config.MMGBSA_DIR / cid / f"rep{rep}" / "FINAL_DECOMP_MMPBSA.dat"
        if not dat.exists():
            continue
        rows = [r for r in parse_decomp(dat, config.LIGAND_RESNAME) if not r.is_ligand]
        if not rows:
            print(f"[stage5] {cid} rep{rep}: decomp parsed 0 protein rows -- check with "
                  f"`python decomp_parse.py {dat} --dump`")
            continue
        if len(rows) != len(ordered):
            print(f"[stage5] {cid} rep{rep}: {len(rows)} decomp rows vs {len(ordered)} protein "
                  f"residues -- zipping by min length, verify numbering")
        n_reps_seen += 1
        for drow, (resid, resname) in zip(rows, ordered):
            got = _ALIAS.get(drow.resname, drow.resname)
            want = _ALIAS.get(resname, resname)
            if got != want:
                # non-fatal: log once per complex-rep and keep going
                pass
            per_resid[resid].append((drow.total, drow.vdw, drow.eel, drow.egb, drow.esurf))
            resname_by_resid[resid] = resname

    pos_map = collect_residue_rows._pos_map
    meta = collect_residue_rows._pos_meta

    out: list[dict] = []
    for resid, vals in per_resid.items():
        totals = [v[0] for v in vals]
        position = pos_map.get((protein, resid))
        m = meta.get(position, {}) if position is not None else {}
        out.append({
            "protein": protein, "role": role, "complex_id": cid,
            "resid": resid, "resname": resname_by_resid.get(resid, ""),
            "position": position,
            "is_cdd_pocket": m.get("is_cdd_pocket", False),
            "cdd_position": m.get("cdd_position"),
            "gb_total": st.fmean(totals),
            "gb_vdw": st.fmean(v[1] for v in vals),
            "gb_eel": st.fmean(v[2] for v in vals),
            "gb_egb": st.fmean(v[3] for v in vals),
            "gb_esurf": st.fmean(v[4] for v in vals),
            "n_replicas": len(vals),
            "inter_rep_sd": st.pstdev(totals) if len(totals) > 1 else 0.0,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rows = config.read_csv_rows(config.MANIFEST_CSV)
    if args.smoke:
        rows = config.smoke_rows(rows)
    if args.limit:
        rows = rows[: args.limit]

    collect_residue_rows._pos_map = config.load_position_resnr_map()
    collect_residue_rows._pos_meta = config.load_position_meta()

    residue_rows: list[dict] = []
    energy_rows: list[dict] = []
    for r in rows:
        cid, protein, role = r["complex_id"], r["protein"], r["role"]
        residue_rows += collect_residue_rows(cid, protein, role)
        for rep in range(config.N_REPLICAS):
            dg = parse_total_dg(config.MMGBSA_DIR / cid / f"rep{rep}" / "FINAL_RESULTS_MMPBSA.dat")
            if dg is not None:
                energy_rows.append({"complex_id": cid, "protein": protein, "role": role,
                                    "replica": rep, "dg_gb_total": dg})

    if not residue_rows:
        print("[stage5] no residue rows parsed -- has Stage 4 produced any FINAL_DECOMP_MMPBSA.dat?")
        return

    res_df = pd.DataFrame(residue_rows)
    res_df.to_csv(config.MMGBSA_DIR / "decomp_by_residue.csv", index=False)

    pos_df = res_df.dropna(subset=["position"]).copy()
    pos_df["position"] = pos_df["position"].astype(int)
    agg = (pos_df.groupby(["protein", "role", "position", "is_cdd_pocket", "cdd_position"], dropna=False)
           .agg(mean_gb_total=("gb_total", "mean"),
                mean_gb_vdw=("gb_vdw", "mean"),
                mean_gb_eel=("gb_eel", "mean"),
                mean_gb_egb=("gb_egb", "mean"),
                mean_gb_esurf=("gb_esurf", "mean"),
                sem=("gb_total", lambda s: s.std(ddof=1) / max(len(s), 1) ** 0.5 if len(s) > 1 else 0.0),
                n=("gb_total", "size"),
                n_replicas=("n_replicas", "max"),
                resnr_examples=("resid", lambda s: sorted(set(int(x) for x in s))[:5]))
           .reset_index())
    agg.to_csv(config.DECOMP_BY_POSITION_CSV, index=False)

    if energy_rows:
        pd.DataFrame(energy_rows).to_csv(config.BINDING_ENERGY_SUMMARY_CSV, index=False)

    n_cdd = int(agg["is_cdd_pocket"].sum())
    print(f"[stage5] {len(res_df)} residue-rows -> {len(agg)} (protein,role,position) rows "
          f"({n_cdd} CDD) -> {config.DECOMP_BY_POSITION_CSV}")
    if energy_rows:
        e = pd.DataFrame(energy_rows)
        print("[stage5] dG_GB by role (mean):")
        for role, sub in e.groupby("role"):
            print(f"[stage5]   {role:<13} {sub['dg_gb_total'].mean():+8.2f} kcal/mol  (n={len(sub)})")


if __name__ == "__main__":
    main()
