"""
mmgbsa/src/make_manifest.py
===========================
Stage 1: the per-complex work list for the MM-GBSA pilot.

Source of truth for which pose to score is redocking/'s Stage 8 output,
redocking/results/comparison/good_pose_representative.csv -- one row per
redocked complex (143 of them: 30 importer, 107 non_importer, 6 ambiguous),
each naming the `caprieval_rank` of that complex's GMM-selected good pose
(usually rank 1; RESULTS.md notes only 2/143 differ). For each row we resolve
the actual model file:

    redocking/results/haddock_runs/<complex_id>/<final>_caprieval/capri_ss.tsv
        -> row with caprieval_rank == <rank>
        -> ../4_flexref/flexref_<N>.pdb  (+ HADDOCK3's in-place .gz)

and join `form` / `ca_cluster` from redocking/data/manifest.csv.

Any complex whose HADDOCK3 run never produced a usable flexref model (RESULTS.md
records ~2 deterministic CNS [flexref] crashes) is dropped with a printed
reason -- it will simply be absent from data/manifest.csv, same convention
redocking/src/make_manifest.py uses.

Output: mmgbsa/data/manifest.csv
  columns: complex_id, protein, role, form, ca_cluster, caprieval_rank,
           haddock_score, n_good_poses, pose_pdb
"""
from __future__ import annotations

import csv
from pathlib import Path

import config


def main() -> None:
    if not config.GOOD_POSE_REPRESENTATIVE_CSV.exists():
        raise FileNotFoundError(
            f"{config.GOOD_POSE_REPRESENTATIVE_CSV} not found -- run redocking/ Stage 8 "
            f"(redocking/src/pose_pocket_engagement.py) first."
        )

    rep_rows = config.read_csv_rows(config.GOOD_POSE_REPRESENTATIVE_CSV)
    redock_meta = {r["complex_id"]: r for r in config.read_csv_rows(config.REDOCKING_MANIFEST_CSV)}

    kept: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for r in rep_rows:
        cid = r["complex_id"]
        run_dir = config.HADDOCK_RUNS_DIR / cid
        if not run_dir.is_dir():
            skipped.append((cid, "no haddock_runs/ dir"))
            continue
        rank = int(r["caprieval_rank"])
        pose = config.model_path_for_rank(run_dir, rank)
        if pose is None:
            skipped.append((cid, f"no flexref model for caprieval_rank={rank} (CNS [flexref] failure?)"))
            continue
        meta = redock_meta.get(cid, {})
        kept.append({
            "complex_id": cid,
            "protein": r["protein"],
            "role": r["role"],
            "form": meta.get("form", ""),
            "ca_cluster": meta.get("ca_cluster", ""),
            "caprieval_rank": rank,
            "haddock_score": r.get("haddock_score", ""),
            "n_good_poses": r.get("n_good_poses", ""),
            "pose_pdb": str(pose.resolve().relative_to(config.PIPELINE_ROOT.resolve())),
        })

    config.MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
        w.writeheader()
        w.writerows(kept)

    n_by_role: dict[str, int] = {}
    for k in kept:
        n_by_role[k["role"]] = n_by_role.get(k["role"], 0) + 1
    print(f"[stage1] {len(kept)} complexes -> {config.MANIFEST_CSV}")
    for role, n in sorted(n_by_role.items()):
        print(f"[stage1]   {role:<13} {n}")
    if skipped:
        print(f"[stage1] skipped {len(skipped)}:")
        for cid, why in skipped:
            print(f"[stage1]   {cid}: {why}")


if __name__ == "__main__":
    main()
