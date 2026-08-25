"""
redocking/src/compare_to_abcfold.py
======================================
Stage 7: the actual question this whole pipeline exists to answer.

- **Importer complex**: does HADDOCK3's physics-based redocking converge on
  a GA1 pose similar to what ABCfold's ab initio cofolding predicted, or a
  different one? Computed as ligand heavy-atom RMSD between HADDOCK3's
  top-ranked model and the original ABCfold holoform pose, AFTER
  superposing the two receptors' C-alpha atoms (HADDOCK3's flexref step
  lets the backbone move slightly, so ligand RMSD must be computed in a
  common receptor frame, not assumed already-aligned).
- **Non-importer complexes**: there is no ABCfold ligand pose to compare
  against (apoform). The question instead is whether HADDOCK3 found ANY
  stable pose engaging the CDD-annotated pocket at all -- computed as
  ligand-heavy-atom-to-CDD-active-residue contacts (<= 4.5 A) for each
  ranked cluster representative. Low/no pocket engagement across clusters
  is consistent with (not proof of) the non-importer classification;
  strong engagement would be the more surprising, worth-investigating
  result.

Reads HADDOCK3's own `capri_ss.tsv`/`capri_clt.tsv` from the run's last
`*_caprieval/` step directory (bonvinlab docs: capri_ss.tsv is per-model,
ranked by HADDOCK score, with a cluster_id + cluster_ranking column once
clustering has run upstream; capri_clt.tsv is the per-cluster aggregate).

**Not yet run against a real HADDOCK3 output directory** -- the ligand
chain-id assumption below (molecule 2 in `molecules=[receptor, ligand]` ->
chain "B", no explicit segid override in make_haddock_cfg.py's template)
needs confirming against an actual run before trusting this script's
numbers; fix `LIGAND_CHAIN_HADDOCK` below if HADDOCK3 assigns something
else.

Output: results/comparison/<complex_id>_comparison.json + a combined
results/comparison/summary.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import gemmi
import numpy as np

import config

LIGAND_CHAIN_HADDOCK = "B"  # UNVERIFIED -- see module docstring
POCKET_CONTACT_CUTOFF = 4.5  # Angstrom, ligand heavy atom <-> CDD active-residue heavy atom


def _find_final_caprieval_dir(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*_caprieval"), key=lambda p: int(p.name.split("_")[0]))
    if not candidates:
        raise FileNotFoundError(f"No *_caprieval/ step directory under {run_dir} -- did the HADDOCK3 "
                                 f"run actually complete? Check {run_dir}/../_slurm_logs/ for errors.")
    return candidates[-1]


def _read_capri_ss(caprieval_dir: Path) -> list[dict]:
    tsv_path = caprieval_dir / "capri_ss.tsv"
    with tsv_path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def top_ranked_model(run_dir: Path) -> tuple[Path, dict]:
    """Best-HADDOCK-score model (cluster rank 1, in-cluster rank 1) from
    the run's final caprieval step."""
    caprieval_dir = _find_final_caprieval_dir(run_dir)
    rows = _read_capri_ss(caprieval_dir)
    if not rows:
        raise ValueError(f"{caprieval_dir / 'capri_ss.tsv'} is empty")
    best = min(rows, key=lambda r: float(r["score"]))
    model_path = Path(rows[0].get("model", ""))
    if not model_path.is_absolute():
        model_path = caprieval_dir / model_path
    return model_path, best


def all_cluster_representatives(run_dir: Path) -> list[tuple[Path, dict]]:
    caprieval_dir = _find_final_caprieval_dir(run_dir)
    rows = _read_capri_ss(caprieval_dir)
    reps = [r for r in rows if r.get("cluster-ranking", r.get("cluster_ranking", "")) in ("1", 1)]
    out = []
    for r in reps:
        model_path = Path(r["model"])
        if not model_path.is_absolute():
            model_path = caprieval_dir / model_path
        out.append((model_path, r))
    return out


def _chain_ca_coords(structure: gemmi.Structure, chain_id: str) -> dict[int, np.ndarray]:
    out = {}
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            for res in chain:
                ca = res.find_atom("CA", "\0")
                if ca is not None:
                    out[res.seqid.num] = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
        break
    return out


def _kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R, t) such that mobile @ R + t ~= target (least-squares)."""
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(mobile_c.T @ target_c)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = u @ np.diag([1, 1, d]) @ vt
    t = target.mean(axis=0) - mobile.mean(axis=0) @ r
    return r, t


def _ligand_heavy_coords(structure: gemmi.Structure, chain_id: str) -> np.ndarray:
    coords = []
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            for res in chain:
                for atom in res:
                    if atom.element.name != "H":
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        break
    return np.array(coords)


def compare_importer(complex_id: str, haddock_model_pdb: Path, abcfold_cif: Path) -> dict:
    haddock_st = gemmi.read_structure(str(haddock_model_pdb))
    haddock_st.setup_entities()
    abcfold_st = gemmi.read_structure(str(abcfold_cif))
    abcfold_st.setup_entities()

    haddock_ca = _chain_ca_coords(haddock_st, config.PROTEIN_CHAIN)
    abcfold_ca = _chain_ca_coords(abcfold_st, config.PROTEIN_CHAIN)
    shared = sorted(set(haddock_ca) & set(abcfold_ca))
    if len(shared) < 10:
        raise ValueError(f"{complex_id}: only {len(shared)} shared C-alpha residues between HADDOCK3 model "
                          f"and ABCfold CIF -- chain-id/numbering mismatch, check LIGAND_CHAIN_HADDOCK "
                          f"and config.PROTEIN_CHAIN assumptions.")
    mobile = np.array([haddock_ca[i] for i in shared])
    target = np.array([abcfold_ca[i] for i in shared])
    r, t = _kabsch_superpose(mobile, target)
    receptor_rmsd = float(np.sqrt(np.mean(np.sum((mobile @ r + t - target) ** 2, axis=1))))

    haddock_ligand = _ligand_heavy_coords(haddock_st, LIGAND_CHAIN_HADDOCK)
    abcfold_ligand = _ligand_heavy_coords(abcfold_st, "L")  # see rescoring/src/config.py's LIGAND_CHAIN
    haddock_ligand_superposed = haddock_ligand @ r + t
    if len(haddock_ligand) != len(abcfold_ligand):
        raise ValueError(f"{complex_id}: ligand heavy-atom count mismatch, HADDOCK3={len(haddock_ligand)} "
                          f"vs ABCfold={len(abcfold_ligand)} -- atom order must match for a meaningful RMSD "
                          f"(both derive from the same GA1 SMILES atom order, but confirm before trusting this).")
    ligand_rmsd = float(np.sqrt(np.mean(np.sum((haddock_ligand_superposed - abcfold_ligand) ** 2, axis=1))))

    return {
        "complex_id": complex_id, "role": "importer",
        "receptor_ca_superposition_rmsd": receptor_rmsd,
        "ligand_rmsd_vs_abcfold_pose": ligand_rmsd,
    }


def compare_non_importer(complex_id: str, protein: str, run_dir: Path) -> dict:
    active_residues = set(config.load_cdd_residues(protein))
    reps = all_cluster_representatives(run_dir)
    cluster_results = []
    for model_path, row in reps:
        st = gemmi.read_structure(str(model_path))
        st.setup_entities()
        ligand_coords = _ligand_heavy_coords(st, LIGAND_CHAIN_HADDOCK)

        contacted_active = set()
        for model in st:
            for chain in model:
                if chain.name != config.PROTEIN_CHAIN:
                    continue
                for res in chain:
                    if res.seqid.num not in active_residues:
                        continue
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                        if np.any(np.linalg.norm(ligand_coords - p, axis=1) <= POCKET_CONTACT_CUTOFF):
                            contacted_active.add(res.seqid.num)
            break
        cluster_results.append({
            "cluster_id": row.get("cluster-id", row.get("cluster_id")),
            "haddock_score": float(row["score"]),
            "n_active_residues_contacted": len(contacted_active),
            "n_active_residues_total": len(active_residues),
        })

    return {"complex_id": complex_id, "role": "non_importer", "clusters": cluster_results}


def main() -> None:
    with config.MANIFEST_CSV.open() as f:
        rows = list(csv.DictReader(f))

    summary_rows = []
    for row in rows:
        complex_id = row["complex_id"]
        run_dir = config.HADDOCK_RUNS_DIR / complex_id
        out_path = config.COMPARISON_DIR / f"{complex_id}_comparison.json"

        if row["role"] == "importer":
            model_path, _ = top_ranked_model(run_dir)
            abcfold_cif = config.PIPELINE_ROOT / row["receptor_cif"]
            result = compare_importer(complex_id, model_path, abcfold_cif)
            summary_rows.append({"complex_id": complex_id, "role": "importer",
                                  "ligand_rmsd_vs_abcfold_pose": result["ligand_rmsd_vs_abcfold_pose"]})
        else:
            result = compare_non_importer(complex_id, row["protein"], run_dir)
            best_cluster = min(result["clusters"], key=lambda c: c["haddock_score"]) if result["clusters"] else {}
            summary_rows.append({"complex_id": complex_id, "role": "non_importer",
                                  "best_cluster_pocket_contacts": best_cluster.get("n_active_residues_contacted", "")})

        config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"{complex_id}: wrote {out_path}")

    summary_csv = config.COMPARISON_DIR / "summary.csv"
    fieldnames = sorted({k for r in summary_rows for k in r})
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
