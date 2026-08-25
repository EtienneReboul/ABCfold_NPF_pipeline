"""
redocking/src/prep_ligand_topology.py
========================================
Stage 1: CNS-format topology/parameter files for GA1, via the actual
BioExcel Building Block for this step -- `biobb_chemistry.acpype.
acpype_params_cns.AcpypeParamsCNS` (a Python-wrapped acpype run producing
CNS/XPLOR-format output directly: .top, .par, .inp, .pdb). Not the raw
acpype CLI -- the whole point of using BioExcel building blocks here is
the standardized biobb Python API (in/out paths + a `properties` dict),
consistent with how the sibling `biobb_wf_haddock` tutorial chains biobb
steps. Route: OpenBabel standardizes/protonates data/ga1_from_ga3.sdf
(build_ga1_from_ga3.py's output) into a plain ligand PDB, then
AcpypeParamsCNS turns that into CNS topology + parameters in one call.

**Known risk, flagged before writing this** (see redocking plan): acpype's
CNS output residue/atom-naming conventions are not guaranteed to match
what HADDOCK3's CNS topology parser (`[topoaa]`) expects out of the box --
biobb's own output/input contract (fixed `output_path_*` filenames) is
also a separate potential mismatch point against what `[topoaa]` looks
for. `validate_against_topoaa` below is not optional cleanup -- it's the
actual test of whether this stage worked, run BEFORE this topology is
ever used in a real docking config. Expect to iterate once a real
mismatch is seen (same category of fix as rescoring/src/prep_ligand.py's
ADD_RING renumbering for PyRosetta params -- don't assume the first
AcpypeParamsCNS run is directly usable).

Fallback: HADDOCK3's own `[topoaa]` `autotoppar = true` mode generates
ligand topology directly from a plain ligand PDB, no acpype step -- use
make_haddock_cfg.py's `--autotoppar` flag to bypass this stage entirely
for a fast pilot cross-check if the acpype route stalls.

Requires: openbabel, biobb_chemistry -- envs/redocking.yaml.

Output: ligand_topology/GA1_cns.top, ligand_topology/GA1_cns.param.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import config


def standardize_with_openbabel(sdf_path: Path, out_pdb: Path) -> Path:
    """obabel: standardize + add explicit hydrogens, SDF -> PDB (acpype's
    expected input format)."""
    obabel = shutil.which("obabel")
    if obabel is None:
        raise RuntimeError("obabel not found on PATH -- activate the redocking conda env (envs/redocking.yaml).")
    subprocess.run([obabel, "-isdf", str(sdf_path), "-opdb", "-O", str(out_pdb), "-h"], check=True)
    return out_pdb


def run_acpype_cns(ligand_pdb: Path, out_dir: Path, net_charge: int = 0, charge_method: str = "bcc") -> tuple[Path, Path]:
    """biobb_chemistry's AcpypeParamsCNS building block: ligand PDB -> CNS
    topology (.top) + parameters (.par), AMBER/GAFF-derived AM1-BCC partial
    charges -- same antechamber/AM1-BCC charge machinery
    rescoring/src/sanitize_for_chimerax.py's ChimeraX dock-prep path
    already relies on for this codebase's ligands, so this isn't a new
    external dependency conceptually, just a new conda env."""
    from biobb_chemistry.acpype.acpype_params_cns import AcpypeParamsCNS

    out_dir.mkdir(parents=True, exist_ok=True)
    out_par = out_dir / "GA1.par"
    out_inp = out_dir / "GA1.inp"
    out_top = out_dir / "GA1.top"
    out_pdb = out_dir / "GA1_acpype.pdb"

    AcpypeParamsCNS(
        input_path=str(ligand_pdb),
        output_path_par=str(out_par),
        output_path_inp=str(out_inp),
        output_path_top=str(out_top),
        output_path_pdb=str(out_pdb),
        properties={"basename": "GA1", "charge": net_charge, "atom_type": charge_method},
    ).launch()

    if not out_top.exists() or not out_par.exists():
        raise FileNotFoundError(f"AcpypeParamsCNS did not produce {out_top} / {out_par} -- check its log "
                                 f"in {out_dir} for the underlying acpype/antechamber failure.")
    return out_top, out_par


def validate_against_topoaa(top_path: Path, param_path: Path, ligand_pdb: Path) -> None:
    """Run a standalone haddock3 [topoaa] on the ligand alone, BEFORE it's
    ever used in a real docking config -- catches CNS naming/format
    mismatches cheaply. See module docstring: this is the real test of
    whether Stage 1 worked, not optional cleanup."""
    haddock3 = shutil.which("haddock3")
    if haddock3 is None:
        raise RuntimeError("haddock3 not found on PATH -- activate the redocking conda env (envs/redocking.yaml).")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cfg_path = tmp_dir / "validate_topoaa.cfg"
        cfg_path.write_text(f"""
run_dir = "{tmp_dir / 'run'}"
molecules = ["{ligand_pdb}"]

[topoaa]
ligand_top_fname = "{top_path}"
ligand_param_fname = "{param_path}"
""")
        result = subprocess.run([haddock3, str(cfg_path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Standalone [topoaa] validation FAILED for {top_path.name}/{param_path.name} -- "
                f"this is the CNS naming/format mismatch flagged in the module docstring, fix it "
                f"before using this topology in a real docking run.\n--- haddock3 stdout ---\n"
                f"{result.stdout}\n--- stderr ---\n{result.stderr}"
            )


def main() -> None:
    if not config.GA1_FROM_GA3_SDF.exists():
        raise FileNotFoundError(f"{config.GA1_FROM_GA3_SDF} not found -- run build_ga1_from_ga3.py first.")

    work_dir = config.DATA_DIR / "_cache" / "ligand_topology"
    work_dir.mkdir(parents=True, exist_ok=True)
    ligand_pdb = standardize_with_openbabel(config.GA1_FROM_GA3_SDF, work_dir / "GA1_standardized.pdb")

    top_src, param_src = run_acpype_cns(ligand_pdb, work_dir / "acpype_out")

    config.LIGAND_TOPOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(top_src, config.GA1_CNS_TOP)
    shutil.copy(param_src, config.GA1_CNS_PARAM)

    validate_against_topoaa(config.GA1_CNS_TOP, config.GA1_CNS_PARAM, ligand_pdb)
    print(f"OK: {config.GA1_CNS_TOP} / {config.GA1_CNS_PARAM} validated against a standalone [topoaa] run.")


if __name__ == "__main__":
    main()
