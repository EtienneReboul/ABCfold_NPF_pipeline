"""
rescoring/src/relief.py
==========================
Clash relief: score the raw pose, then a light coordinate-constrained
FastRelax to relieve clashes without drifting from the co-folded geometry.

Ported from NPF_pocket_pipeline/rescoring/src/relief.py — generic already
(only ever touches the staged PDB's ligand residue and its RELIEF_RADIUS_A
neighborhood, regardless of what ligand or backend produced the pose), with
one necessary change to init_pyrosetta: the sibling project only ever had
ONE ligand, so it only ever needed to load one `-extra_res_fa` params file,
once, per process. This pipeline's `run_batch.py` workers process complexes
across MULTIPLE ligands (manifest rows aren't grouped by ligand before
being handed to the worker pool) — and `pyrosetta.init()` only registers
`-extra_res_fa` residue types passed in the FIRST call in a process; a
SECOND call naming a different params file is silently ignored (confirmed
by hand: PyRosetta doesn't error, it falls back to auto-perceiving a
generic "pdb_<resname>" residue type from the raw PDB coordinates instead —
exactly the naive proximity-based bond-order guessing ligand_fix.py exists
to avoid, just reappearing silently one layer downstream). So every ligand
this process might ever need to score must be passed together, in ONE
`-extra_res_fa <path1> <path2> ...` call, before scoring anything — see
`init_pyrosetta` below, which takes the FULL list rather than one path.

**Restricted to a neighborhood around the ligand, not the whole protein.**
This is a deliberate deviation from a literal "FastRelax the whole pose", and
was found necessary, not just faster: an unrestricted FastRelax on these
never-crystallographically-refined structures blew the total score up
catastrophically in the sibling project's testing (-665 REU -> +36,677 REU)
— isolating the FastRelax stages showed it was full-protein *repacking*
specifically (all residues repacked, no restriction) that was unstable
(-675 -> +1,548 from packing alone; plain minimization alone, by contrast,
was fine and improved the score to -1,645). Restricting packing +
minimization to residues within RELIEF_RADIUS_A of the ligand keeps the
score change modest (no explosion) and is exactly what "light... without
drifting... do not over-minimize" calls for, while also being far cheaper
across thousands of complexes.
"""
from __future__ import annotations

from pathlib import Path

import pyrosetta
from pyrosetta import rosetta

_INITIALIZED = False

RELIEF_RADIUS_A = 10.0  # neighborhood radius around the ligand allowed to repack/minimize


def init_pyrosetta(params_paths, seed: int | None = None) -> None:
    """
    `params_paths`: every ligand's params file this process might ever need
    to score, as one path or an iterable of paths -- ALL of them, not just
    the current complex's own ligand (see module docstring for why: a
    second, different `-extra_res_fa` in a later call is silently ignored,
    not additive). Only the first call in a process actually does anything;
    later calls (even with a different params_paths) are a no-op, same as
    upstream PyRosetta's own init() semantics.

    seed: pass a fixed value for a fully reproducible run (same replica-to-
    replica sequence every time); omit it for normal ensemble variation
    (FastRelax is stochastic, so successive replicas/reruns genuinely differ
    without a fixed seed — that's what n_replicas is for). Note: like every
    other run-level flag here, `-run:jran` only takes effect on this
    process's FIRST call — every complex after the first one a given
    run_batch.py worker processes shares that same RNG stream rather than
    getting its own independently-seeded one. Still deterministic given a
    fixed base seed + worker scheduling, just not per-complex-independent
    the way `run_batch.py`'s `_complex_seed` docstring implies in isolation.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    if isinstance(params_paths, (str, Path)):
        params_paths = [params_paths]
    extra_res_fa = " ".join(str(p) for p in params_paths)
    seed_flags = f"-run:constant_seed -run:jran {seed} " if seed is not None else ""
    pyrosetta.init(
        f"-extra_res_fa {extra_res_fa} "
        f"{seed_flags}"
        "-mute all "
        "-relax:constrain_relax_to_start_coords "
        "-relax:coord_constrain_sidechains "
        "-relax:ramp_constraints true "
        "-no_optH false"
    )
    _INITIALIZED = True


def load_pose(pdb_path) -> pyrosetta.Pose:
    return pyrosetta.pose_from_pdb(str(pdb_path))


def ligand_residue_index(pose: pyrosetta.Pose, ligand_resname: str) -> int:
    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).name3().strip() == ligand_resname:
            return i
    raise ValueError(f"No {ligand_resname} residue found in pose.")


def neighborhood_movemap_and_task(pose: pyrosetta.Pose, ligand_resi: int, radius: float = RELIEF_RADIUS_A):
    """MoveMap (bb+chi) and TaskFactory (repack only), both restricted to
    residues within `radius` of the ligand — see module docstring for why."""
    lig_sel = rosetta.core.select.residue_selector.ResidueIndexSelector(str(ligand_resi))
    nbr_sel = rosetta.core.select.residue_selector.NeighborhoodResidueSelector(lig_sel, radius, True)
    in_neighborhood = nbr_sel.apply(pose)

    movemap = rosetta.core.kinematics.MoveMap()
    for i in range(1, pose.total_residue() + 1):
        if in_neighborhood[i]:
            movemap.set_bb(i, True)
            movemap.set_chi(i, True)

    task_factory = rosetta.core.pack.task.TaskFactory()
    task_factory.push_back(rosetta.core.pack.task.operation.RestrictToRepacking())
    prevent = rosetta.core.pack.task.operation.PreventRepacking()
    for i in range(1, pose.total_residue() + 1):
        if not in_neighborhood[i]:
            prevent.include_residue(i)
    task_factory.push_back(prevent)

    return movemap, task_factory


def light_relax(pose: pyrosetta.Pose, sfxn, ligand_resi: int, cycles: int = 1) -> None:
    """
    Coordinate-constrained FastRelax restricted to the ligand's neighborhood
    (see module docstring). The coordinate-constraint behavior (constrain to
    start coords, ramp down) is controlled by the `-relax:*` flags passed
    once to pyrosetta.init() in init_pyrosetta() above. fa_rep ramping is
    FastRelax's normal default staged repulsive-ramping schedule, no extra
    flag needed.
    """
    movemap, task_factory = neighborhood_movemap_and_task(pose, ligand_resi)
    relax = rosetta.protocols.relax.FastRelax(sfxn, cycles)
    relax.set_movemap(movemap)
    relax.set_task_factory(task_factory)
    relax.apply(pose)


def relieve_clashes(pdb_path, params_paths, ligand_resname: str, n_replicas: int = 1,
                     relax_cycles: int = 1, seed: int | None = None):
    """
    Score raw -> light relax -> score relaxed, for n_replicas independent
    trajectories (stochastic FastRelax — replicas give an ensemble estimate).

    `params_paths` should be EVERY ligand's params file this process might
    ever need (config.all_params_paths()), not just this complex's own --
    see init_pyrosetta / module docstring for why (only the first
    `-extra_res_fa` in a process actually registers).

    `ligand_resname` is this complex's ligand's own distinct residue code
    (config.ligand_resname(ligand_key)) — required, not defaulted to "LIG",
    since a wrong default here would silently score against whatever
    ligand's residue type happens to already be loaded (see this module's
    docstring / config.py's LIGAND_CODES for why every ligand needs its own
    distinct name).

    Returns: list of dicts, one per replica:
        {replica, fa_rep_raw, total_raw, fa_rep_relaxed, total_relaxed, pose}
    """
    init_pyrosetta(params_paths, seed=seed)
    sfxn = pyrosetta.get_score_function()

    results = []
    for replica in range(n_replicas):
        pose = load_pose(pdb_path)
        ligand_resi = ligand_residue_index(pose, ligand_resname)
        sfxn(pose)
        fa_rep_raw = pose.energies().total_energies()[rosetta.core.scoring.fa_rep]
        total_raw = pose.energies().total_energies()[rosetta.core.scoring.total_score]

        light_relax(pose, sfxn, ligand_resi, cycles=relax_cycles)

        sfxn(pose)
        fa_rep_relaxed = pose.energies().total_energies()[rosetta.core.scoring.fa_rep]
        total_relaxed = pose.energies().total_energies()[rosetta.core.scoring.total_score]

        results.append({
            "replica": replica,
            "fa_rep_raw": fa_rep_raw,
            "total_raw": total_raw,
            "fa_rep_relaxed": fa_rep_relaxed,
            "total_relaxed": total_relaxed,
            "pose": pose,
        })
    return results
