#!/usr/bin/env python3
"""
scripts/generate_abcfold_visuals.py
=====================================
Regenerate ABCfold's own results-table + PAE-viewer output page
(index.html, .plots/, output_models/, open_output.py) for an ALREADY
COMPLETED protein x form run, without re-running any of the 6 backends.

Why this exists: worflows/processing/submit_abcfold.sh always passes
--no_visuals (and --no_server) so SLURM compute nodes never try to open a
browser or serve HTTP — but abcfold.abcfold.run() returns IMMEDIATELY when
--no_visuals is set, before it ever builds outputs, plots, or index.html.
ABCfold also has no "resume / regenerate visuals only" mode (per its own
docs, one `abcfold` call either completes or it doesn't) — the only
documented way to get the visuals page is a from-scratch --override run,
which would redo all 6 backends' GPU work just to get a webpage.

This script instead reconstructs the same AlphafoldOutput/BoltzOutput/.../
RosettafoldOutput wrapper objects abcfold's own run() builds mid-flight, by
pointing them at the ALREADY-WRITTEN backend result directories, then calls
abcfold's own (unmodified) plots()/render_template()/output_open_html_script()
to produce the same output page — using abcfold's installed library code
throughout, not a reimplementation of its plotting/scoring logic.

Two non-obvious ABCfold internals this script has to work around (both
confirmed 2026-08-01 against a completed submit_abcfold.sh run — read
abcfold/output/{boltz,chai,openfold3,protenix,rosettafold3}.py before
"fixing" either of these if a future abcfold version changes them):

1. Each XOutput.__init__ (Boltz/Chai-1/OpenFold3/Protenix/RosettaFold3 —
   AlphaFold3 is the only exception) unconditionally MOVES its own
   backend_results_*/ directories on disk into a wrapper directory
   {backend}_{name}/ that it creates. This isn't optional and isn't
   idempotent: constructing XOutput a *second* time against an
   already-wrapped directory nests a second wrapper one level deeper
   instead of finding anything (observed: .../boltz_NAME/boltz_NAME/...).
   Since these Output objects get constructed once already during the real
   prediction run (that happens regardless of --no_visuals — only the
   plotting/HTML step is skipped), the wrapper already exists by the time
   this script runs. unwrap_backend_dirs() below undoes the wrapping first
   (moving result dirs back to directly under output_dir, matching what a
   fresh, never-before-processed run would look like — what XOutput.__init__
   actually expects as input), so the constructor's own rename recreates the
   exact same single-level wrapper instead of nesting a new one.

2. Each XOutput.pae_to_af3() converts that backend's native PAE data to
   AF3's PAE JSON schema and writes it via pae.to_file(out_name), where
   out_name is the ORIGINAL raw PAE file's own path — i.e. it overwrites the
   raw PAE data in place. Since this already happened once during the real
   run, the "pae_*" files on disk (.npz for most backends, .npy for Chai-1)
   are actually AF3-PAE JSON text with a stale binary extension, and
   np.load() on them raises UnpicklingError.
   patch_af3_pae_reentrancy() below makes NpzFile treat JSON-content files as
   already-converted data, and makes each Af3Pae.from_X() classmethod pass
   already-AF3-shaped dicts straight through instead of re-deriving them
   (which would assume raw, backend-native input and produce garbage).

Usage:
    python scripts/generate_abcfold_visuals.py \\
        --output-dir results/abcfold/NPF2.12_Q9LFX9__apo
"""

import argparse
import configparser
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                    help="Completed ABCfold run directory (results/abcfold/<protein>)")
    p.add_argument("--input-json", default=None,
                    help="fold_input json abcfold saved (default: <output-dir>/abc_fold_input.resolved.json)")
    p.add_argument("--port", type=int, default=8000,
                    help="Port baked into the generated open_output.py")
    return p.parse_args()


def unwrap_backend_dirs(output_dir: Path, wrapper_prefix: str, result_pattern: str):
    """Undo the wrapping XOutput.__init__ already did once during the real
    prediction run (see module docstring point 1), moving result_pattern*
    dirs back to directly under output_dir — where XOutput.__init__ actually
    expects to find them (matching a fresh, never-before-processed run) —
    so its own rename recreates the same single-level wrapper instead of
    nesting a new one on top."""
    moved = []
    for wrapper in sorted(output_dir.glob(f"{wrapper_prefix}_*")):
        if not wrapper.is_dir():
            continue
        for result_dir in sorted(wrapper.glob(result_pattern)):
            target = output_dir / result_dir.name
            if not target.exists():
                result_dir.rename(target)
            moved.append(target)
    if moved:
        return moved
    # Nothing wrapped yet (e.g. a genuinely fresh directory) — fall back to
    # a direct glob, matching abcfold.abcfold.run()'s own assumption.
    return sorted(output_dir.glob(result_pattern))


def clean_chai_pae_copies(chai_dirs):
    """ChaiOutput.process_chai_output() identifies the one shared raw PAE
    file per seed (pae_scores.npy) by a bare 'startswith("pae_scores")'
    check — which also matches the per-model COPIES of that same file
    (pae_scores_model_N.npy) it creates as a side effect of its own
    processing. On a second construction it can pick one of those stale
    copies instead of the master, then try to shutil.copy the master onto
    that same file, raising SameFileError (confirmed 2026-08-01). The
    copies are always exact, always-regenerable duplicates of
    pae_scores.npy — never authoritative — so just delete them; ChaiOutput
    recreates them fresh from the untouched master during processing."""
    for chai_dir in chai_dirs:
        for stale_copy in chai_dir.glob("pae_scores_model_*.npy"):
            stale_copy.unlink()


def patch_af3_pae_reentrancy():
    """See module docstring point 2. Safe no-op for genuinely-fresh (never
    before converted) PAE data — only activates when a PAE file's content is
    already AF3-PAE-shaped JSON."""
    from abcfold.output.file_handlers import NpyFile, NpzFile
    from abcfold.output.utils import AF3TEMPLATE, Af3Pae

    orig_load_npz = NpzFile.load_npz_file

    def patched_load_npz(self):
        with open(self.npz_file, "rb") as f:
            head = f.read(1)
        if head == b"{":
            return json.loads(Path(self.npz_file).read_text())
        return orig_load_npz(self)

    NpzFile.load_npz_file = patched_load_npz

    # Chai-1's PAE files are .npy (ndarray), not .npz (dict) — same
    # already-converted-in-place issue, same fix.
    orig_load_npy = NpyFile.load_npy_file

    def patched_load_npy(self):
        with open(self.npy_file, "rb") as f:
            head = f.read(1)
        if head == b"{":
            return json.loads(Path(self.npy_file).read_text())
        return orig_load_npy(self)

    NpyFile.load_npy_file = patched_load_npy

    template_keys = set(AF3TEMPLATE.keys())

    def make_passthrough(orig_from_x):
        def wrapper(scores, cif_file):
            if isinstance(scores, dict) and template_keys <= scores.keys():
                return Af3Pae(scores)
            return orig_from_x(scores, cif_file)
        return wrapper

    for name in ("from_boltz", "from_chai1", "from_openfold3",
                 "from_protenix", "from_rosettafold3", "from_alphafold3"):
        setattr(Af3Pae, name, make_passthrough(getattr(Af3Pae, name)))


def collect_models(output_obj, method, indicies, index_counter, plot_dict, output_dir,
                    get_model_data, insert_none_by_minus_one, skip_negative_idx):
    models = []
    if output_obj is None:
        return models, index_counter
    score_key = "summary" if method == "AlphaFold3" else (
        "json" if method == "Boltz" else "scores"
    )
    for seed in output_obj.output.keys():
        for idx in output_obj.output[seed].keys():
            if skip_negative_idx and idx < 0:
                continue
            model = output_obj.output[seed][idx]["cif"]
            model.check_clashes()
            score_file = output_obj.output[seed][idx][score_key]
            plddt = model.residue_plddts
            pae = output_obj.output[seed][idx]["af3_pae"]
            if len(indicies) > 0:
                plddt = insert_none_by_minus_one(indicies[index_counter], plddt)
            index_counter += 1
            models.append(get_model_data(
                model, plot_dict, method, plddt, pae, score_file, output_dir,
                affinity_scores={},
            ))
    return models, index_counter


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    input_json = Path(args.input_json) if args.input_json else output_dir / "abc_fold_input.resolved.json"

    if not input_json.exists():
        sys.exit(f"Input json not found: {input_json}")

    import abcfold.abcfold as abcfold_mod
    from abcfold.html.html_utils import (get_all_cif_files, get_model_data,
                                         get_model_sequence_data,
                                         output_open_html_script, plots,
                                         render_template)
    from abcfold.output.alphafold3 import AlphafoldOutput
    from abcfold.output.boltz import BoltzOutput
    from abcfold.output.chai import ChaiOutput
    from abcfold.output.file_handlers import superpose_models
    from abcfold.output.openfold3 import OpenfoldOutput
    from abcfold.output.protenix import ProtenixOutput
    from abcfold.output.rosettafold3 import RosettafoldOutput
    from abcfold.output.utils import get_gap_indicies, insert_none_by_minus_one

    HTML_DIR = abcfold_mod.HTML_DIR
    HTML_TEMPLATE = abcfold_mod.HTML_TEMPLATE
    PLOTS_DIR = abcfold_mod.PLOTS_DIR

    patch_af3_pae_reentrancy()

    input_params = json.loads(input_json.read_text())
    name = input_params.get("name")
    if not name:
        sys.exit(f"{input_json} has no 'name' field")

    output_dir.joinpath(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # Same config abcfold itself reads/creates at ~/.abcfold_config.ini
    config_file = Path.home() / ".abcfold_config.ini"
    config = configparser.ConfigParser()
    if config_file.exists():
        config.read(str(config_file))
    rt_config = {}
    for section in config.sections():
        rt_config.update(dict(config.items(section)))

    outputs = []
    programs_run = []
    ao = bo = co = oo = po = ro = None

    af3_dirs = sorted(output_dir.glob("alphafold3_*"))
    if af3_dirs:
        ao = AlphafoldOutput(af3_dirs[0], input_params, name)
        outputs.append(ao)
        programs_run.append("AlphaFold3")

    boltz_dirs = unwrap_backend_dirs(output_dir, "boltz", "boltz_results*")
    if boltz_dirs:
        bo = BoltzOutput(boltz_dirs, input_params, name)
        outputs.append(bo)
        programs_run.append("Boltz")

    chai_dirs = unwrap_backend_dirs(output_dir, "chai1", "chai_output*")
    if chai_dirs:
        clean_chai_pae_copies(chai_dirs)
        co = ChaiOutput(chai_dirs, input_params, name, rt_config)
        outputs.append(co)
        programs_run.append("Chai-1")

    openfold_dirs = unwrap_backend_dirs(output_dir, "openfold", "openfold_results*")
    if openfold_dirs:
        oo = OpenfoldOutput(openfold_dirs, input_params, name)
        outputs.append(oo)
        programs_run.append("OpenFold3")

    protenix_dirs = unwrap_backend_dirs(output_dir, "protenix", "protenix_results*")
    if protenix_dirs:
        po = ProtenixOutput(protenix_dirs, input_params, name)
        outputs.append(po)
        programs_run.append("Protenix")

    rosettafold_dirs = unwrap_backend_dirs(output_dir, "rosettafold", "rosettafold_results*")
    if rosettafold_dirs:
        ro = RosettafoldOutput(rosettafold_dirs, input_params, name)
        outputs.append(ro)
        programs_run.append("RosettaFold3")

    if not outputs:
        sys.exit(f"No completed backend output found under {output_dir}")

    print(f"[visuals] {name}: found {', '.join(programs_run)}")

    plot_dict = plots(outputs, output_dir.joinpath(PLOTS_DIR), make_pae_plots=True)

    cif_models = [
        cif_file
        for cif_list in get_all_cif_files(outputs).values()
        for cif_file in cif_list
    ]
    indicies = get_gap_indicies(*cif_models)
    index_counter = 0

    combined_models = []
    for output_obj, method, skip_neg in (
        (ao, "AlphaFold3", False),
        (bo, "Boltz", False),
        (co, "Chai-1", True),
        (oo, "OpenFold3", True),
        (po, "Protenix", True),
        (ro, "RosettaFold3", True),
    ):
        models, index_counter = collect_models(
            output_obj, method, indicies, index_counter, plot_dict, output_dir,
            get_model_data, insert_none_by_minus_one, skip_neg,
        )
        combined_models.extend(models)

    os.makedirs(output_dir.joinpath("output_models"), exist_ok=True)
    output_models = []
    name_prefix = {
        "AlphaFold3": "af3", "Boltz": "boltz", "Chai-1": "chai",
        "OpenFold3": "openfold", "Protenix": "protenix", "RosettaFold3": "rosettafold",
    }
    for model in combined_models:
        cif_file = output_dir.joinpath(model["model_path"])
        output_name = f"{name_prefix[model['model_source']]}_model_{model['model_id'][-1]}.cif"
        dest = output_dir.joinpath("output_models").joinpath(output_name)
        shutil.copy(cif_file, dest)
        output_models.append(dest)

    if len(output_models) > 1:
        superpose_models(output_models)

    sequence_data = get_model_sequence_data(cif_models)
    sequence = "".join(sequence_data.values())
    chain_data = {}
    ref = 0
    for key, seq in sequence_data.items():
        chain_data["Chain " + key] = (ref, len(seq) + ref - 1)
        ref += len(seq)

    results_dict = {
        "sequence": sequence,
        "models": combined_models,
        "plotly_path": Path(plot_dict["plddt"]).relative_to(output_dir.resolve()).as_posix(),
        "chain_data": chain_data,
        "ccp4cloud": False,
    }
    results_json = json.dumps(results_dict)

    if not output_dir.joinpath(".feature_viewer").exists():
        shutil.copytree(HTML_DIR, output_dir / ".feature_viewer")

    if len(programs_run) > 1:
        programs = ("Structure predictions for: " + ", ".join(programs_run[:-1])
                    + " and " + programs_run[-1])
    else:
        programs = "Structure predictions for: " + programs_run[0]

    html_out = output_dir.joinpath("index.html").resolve()
    render_template(
        HTML_TEMPLATE, html_out,
        abcfold_html_dir=".feature_viewer",
        programs=programs,
        results_json=results_json,
        version=0.1,
    )
    print(f"[visuals] Output page written: {html_out}")

    output_open_html_script(str(output_dir / "open_output.py"), port=args.port)
    print(f"[visuals] open_output.py written. To view:\n  cd {output_dir}\n  python open_output.py")


if __name__ == "__main__":
    main()
