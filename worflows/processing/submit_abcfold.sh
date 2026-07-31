#!/usr/bin/env bash
# =============================================================================
# submit_abcfold.sh — Stage 4: ABCfold structure prediction on the IFB cluster
# =============================================================================
#
# For every pending protein x form (apo/holo), runs ONE `abcfold` call that
# launches AlphaFold3, Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3
# together against the same fold_input.resolved.json
# (https://github.com/rigdenlab/ABCFold) — the whole point of switching away
# from AF3_NPF_pipeline: notebook/tm_conformation_clustering_gibberellin_boltz.ipynb
# (in that repo) showed AF3 alone, even with 30 seeds, does not recover every
# conformation a second model (Boltz-2) finds for the same sequences, so this
# pipeline samples several independent architectures per protein instead of
# just resampling one.
#
# MSA + templates were already resolved locally (worflows/preprocessing/Snakefile,
# scripts/fetch_mmseqs2_msa.py — ABCfold's own `mmseqs2msa` CLI against the
# ColabFold MMseqs2 webserver) and embedded into fold_input.resolved.json, so
# this script never passes --mmseqs2 or --templates: compute nodes need no
# outbound network access for that part. What compute nodes on IFB likely
# DON'T have, unlike AF3_NPF_pipeline's `module load alphafold`: Docker.
# ABCfold's AlphaFold3 backend shells out to `docker run` by default, or
# `singularity exec` if --af3_sif_path is set.
#
# CONFIRMED on IFB (2026-07-31): `module load alphafold/$AF3_MODULE_VERSION`
# is itself just a thin wrapper around Singularity —
#   /shared/software/singularity/wrappers/alphafold/<version>/run_alphafold.py
# is a bash script that does `singularity exec ... <version>.sif run_alphafold.py $@`.
# So no separate AF3 .sif needs to be built: discover_af3_sif() below reads
# that same wrapper script and extracts the .sif path it already uses,
# instead of hardcoding an absolute path that would go stale across module
# version bumps. Set AF3_SIF_PATH manually (or config.yaml's
# abcfold.af3_sif_path) only if this auto-discovery ever fails — e.g. IFB
# changes the wrapper layout, or you're running this on a different cluster.
#
# One-time setup (needs internet — run on a login node, NOT a compute node):
#   Boltz, Chai-1, OpenFold3, Protenix and RosettaFold3 are auto-installed by
#   ABCfold into internal micromamba environments the first time each one
#   runs. Prime that install with `--dry_run` (sets up every selected
#   predictor's env + weights + a --help smoke test, no GPU, no inference):
#     bash submit_abcfold.sh --prime
#
# Prerequisites:
#   - worflows/preprocessing/Snakefile completed (fold_input.resolved.json exists per run)
#   - Run from the pipeline root directory
#   - micromamba available on $PATH (ABCfold requires it to build backend envs)
#   - `module load singularity` works (loaded automatically below)
#   - `bash submit_abcfold.sh --prime` has completed successfully at least once
#
# Flags below (--number_of_models, --num_recycles, --model_params,
# --af3_sif_path, --override, --no_server, --no_visuals) are confirmed
# against abcfold/argparse_utils.py in the ABCFold source (main branch).
# What's NOT confirmed is the exact output directory layout ABCfold writes
# per backend under <output_dir> (abcfold/output/{alphafold3,boltz,chai,
# openfold3,protenix,rosettafold3}.py suggest `alphafold3_<name>/`,
# `boltz_results_<name>/`, `chai_output_<name>/`, `openfold_results_<name>/`,
# `protenix_results_<name>/`, `rosettafold_results_<name>/`, each holding
# *.cif somewhere under it, but this is read from source, not a real run).
# Run --test once and inspect results/abcfold/<protein>/ before scaling up;
# adjust scripts/tm_helix_alignment.py's discover_predictions() if the
# layout doesn't match.
#
# Usage:
#   bash submit_abcfold.sh --prime                     # one-time backend env warm-up (login node)
#   bash submit_abcfold.sh --dry-run                    # show plan only
#   bash submit_abcfold.sh --test                       # submit task 0 only (QoS-safe test)
#   bash submit_abcfold.sh --batch-size 1                # proteins per array task
#   bash submit_abcfold.sh --max-concurrent 5            # max parallel tasks
#   bash submit_abcfold.sh --models abcopr               # which backends (-a-b-c-o-p-r letters)
#   bash submit_abcfold.sh --gres gpu:l40s:1
#
# ABCfold has no documented cache-and-resume split the way run_alphafold.py's
# --norun_inference/--norun_data_pipeline did — one `abcfold` call either
# completes or it doesn't. So unlike submit_af3.sh, a retried run always
# passes --override and recomputes every backend from scratch; only
# proteins with a prediction.done sentinel are skipped entirely.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PARAMS="/shared/bank/alphafold3/current"   # AF3 weights dir (config.yaml abcfold.model_params)
AF3_MODULE_VERSION="3.0.2"                        # config.yaml abcfold.af3_module_version
AF3_SIF_PATH=""                                   # leave empty to auto-discover (see discover_af3_sif below);
                                                   # set explicitly (or config.yaml's abcfold.af3_sif_path) to override
CUDA_MODULE_VERSION="12.9.1"                      # IFB cuda-toolkit module — Protenix needs CUDA_HOME set to
                                                   # build its CUDA extensions; see discover_cuda_home below
MODELS="abcopr"                                   # -a -b -c -o -p -r letters to run together

NUMBER_OF_MODELS=5        # abcfold --number_of_models (config.yaml abcfold.number_of_models)
NUM_RECYCLES=10            # abcfold --num_recycles     (config.yaml abcfold.num_recycles)

# ── Discover the AF3 .sif IFB's `module load alphafold` already wraps ────────
# Confirmed 2026-07-31: `module show alphafold/$AF3_MODULE_VERSION` prepends
# /shared/software/singularity/wrappers/alphafold/$AF3_MODULE_VERSION to PATH,
# and that directory's run_alphafold.py is a bash wrapper whose last line is
# `singularity exec ... <something>.sif run_alphafold.py $@`. Extract that
# .sif token; if it's a bare filename (no leading '/'), resolve it relative to
# the wrapper directory, which is where IFB's module system places it.
discover_af3_sif() {
    local wrapper_dir="/shared/software/singularity/wrappers/alphafold/$AF3_MODULE_VERSION"
    local wrapper="$wrapper_dir/run_alphafold.py"
    if [[ ! -f "$wrapper" ]]; then
        return 1
    fi
    local sif
    sif=$(grep -oE '[^[:space:]]+\.sif' "$wrapper" | head -n1)
    if [[ -z "$sif" ]]; then
        return 1
    fi
    if [[ "$sif" != /* ]]; then
        sif="$wrapper_dir/$sif"
    fi
    if [[ ! -f "$sif" ]]; then
        return 1
    fi
    echo "$sif"
}

if [[ -z "$AF3_SIF_PATH" ]]; then
    if DISCOVERED_SIF=$(discover_af3_sif); then
        AF3_SIF_PATH="$DISCOVERED_SIF"
        echo "[submit_abcfold] Auto-discovered AF3 .sif: $AF3_SIF_PATH"
    fi
fi

# ── Discover CUDA_HOME from IFB's cuda-toolkit module ─────────────────────────
# Confirmed 2026-07-31: `module load cuda-toolkit/$CUDA_MODULE_VERSION` only
# prepends PATH/CPATH — it never sets CUDA_HOME or CUDA_PATH. Protenix's build
# step needs CUDA_HOME pointing at the toolkit root to compile its CUDA
# extensions (observed failure: "OSError: CUDA_HOME environment variable is
# not set"), so derive it the same way discover_af3_sif() derives the .sif
# path: read the modulefile directly rather than requiring `module` state.
discover_cuda_home() {
    local modulefile="/shared/software/modulefiles/cuda-toolkit/$CUDA_MODULE_VERSION"
    if [[ ! -f "$modulefile" ]]; then
        return 1
    fi
    local env_bin env_root targets_root
    env_bin=$(grep -oE '[^[:space:]]+/envs/cuda-toolkit-[^[:space:]]+/bin' "$modulefile" | head -n1)
    if [[ -z "$env_bin" ]]; then
        return 1
    fi
    env_root=$(dirname "$env_bin")
    targets_root=$(find "$env_root/targets" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -n1)
    if [[ -z "$targets_root" || ! -f "$targets_root/include/cuda_runtime_api.h" ]]; then
        # No split targets/<arch> layout found — env_root itself is the best guess.
        echo "$env_root"
        return 0
    fi
    # Conda-forge's cuda-toolkit package splits itself in a way neither half
    # alone satisfies on its own (confirmed 2026-07-31, via Protenix's fused
    # CUDA kernel build failing two different ways):
    #   env_root/{bin/nvcc,nvvm/bin/cicc}   — real nvcc + its cicc front-end,
    #                                          but env_root/include is a
    #                                          generic compiler_compat dir
    #                                          with no CUDA headers at all
    #                                          ("cuda_runtime_api.h: No such
    #                                          file or directory").
    #   targets/<arch>/{include,lib}        — the real CUDA headers/libs, but
    #                                          targets/<arch>/bin/nvcc is a
    #                                          symlink back to env_root/bin/nvcc,
    #                                          and nvcc's own cicc lookup
    #                                          (relative to how it was
    #                                          invoked) misses cicc from
    #                                          there ("cicc: not found").
    # Build a small shim combining the real (non-symlinked) bin/+nvvm/ from
    # env_root with the real include/+lib/ from targets/<arch>, so CUDA_HOME
    # gets a single, flat, self-consistent directory as nvcc expects. Lives
    # under our own project dir since env_root/targets are admin-owned.
    local shim="/shared/projects/npf_abinitio/conda/cuda_home_shim"
    mkdir -p "$shim"
    ln -sfn "$env_root/bin" "$shim/bin"
    ln -sfn "$env_root/nvvm" "$shim/nvvm"
    ln -sfn "$targets_root/include" "$shim/include"
    ln -sfn "$targets_root/lib" "$shim/lib"
    ln -sfn "$targets_root/lib" "$shim/lib64"
    echo "$shim"
}

CUDA_HOME_PATH=""
if DISCOVERED_CUDA_HOME=$(discover_cuda_home); then
    CUDA_HOME_PATH="$DISCOVERED_CUDA_HOME"
    echo "[submit_abcfold] Auto-discovered CUDA_HOME: $CUDA_HOME_PATH"
else
    echo "WARNING: could not auto-discover CUDA_HOME from cuda-toolkit/$CUDA_MODULE_VERSION —"
    echo "         Protenix (-p) will likely fail to build its CUDA extensions."
fi

# Batching — one array task handles BATCH_SIZE protein x form runs, each
# running its own single `abcfold -abcopr ...` call in sequence within the task.
BATCH_SIZE=1
MAX_CONCURRENT=10

# SLURM resources (per array task) — six backends in one task needs more
# memory/time headroom than the AF3-only pipeline's per-protein task.
PARTITION="gpu"
CPUS=8
MEM="80G"
GRES="gpu:l40s:1"
TIME=600                  # minutes per task
ACCOUNT=""

FOLD_IN_DIR="data/fold_inputs"
ABCFOLD_OUT_DIR="results/abcfold"
PRIORITY_MANIFEST="$FOLD_IN_DIR/priority_gibberellin.txt"

# ── Parse arguments ───────────────────────────────────────────────────────────
DRY_RUN=false
TEST_MODE=false
PRIME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true; shift ;;
        --prime)
            PRIME=true; shift ;;
        --gres)
            GRES="$2"; shift 2 ;;
        --gres=*)
            GRES="${1#--gres=}"; shift ;;
        --models)
            MODELS="$2"; shift 2 ;;
        --models=*)
            MODELS="${1#--models=}"; shift ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*)
            BATCH_SIZE="${1#--batch-size=}"; shift ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"; shift 2 ;;
        --max-concurrent=*)
            MAX_CONCURRENT="${1#--max-concurrent=}"; shift ;;
        --test)
            TEST_MODE=true; shift ;;
        *)
            echo "ERROR: unknown argument '$1'"
            echo "Usage: bash submit_abcfold.sh [--prime] [--dry-run] [--test] [--gres <profile>]"
            echo "                              [--models abcopr] [--batch-size N]"
            echo "                              [--max-concurrent N]"
            exit 1 ;;
    esac
done

MODEL_FLAG="-${MODELS}"

# ── Prime mode: warm up internal micromamba backend envs (needs internet) ────
if $PRIME; then
    echo "============================================================"
    echo " ABCfold --dry_run priming (backend env + weights setup only)"
    echo " Models  : $MODEL_FLAG"
    echo " Run this on a LOGIN NODE (or any node with outbound internet),"
    echo " never on a compute node — this is the only step that installs"
    echo " Boltz/Chai-1/OpenFold3/Protenix/RosettaFold3."
    echo "============================================================"
    ANY_JSON=$(find "$FOLD_IN_DIR" -maxdepth 2 -name 'fold_input.resolved.json' -print -quit)
    if [[ -z "$ANY_JSON" ]]; then
        echo "ERROR: no fold_input.resolved.json found under $FOLD_IN_DIR." \
             "Run worflows/preprocessing/Snakefile first." >&2
        exit 1
    fi
    module load singularity   # ABCfold shells out to `singularity exec` directly for AF3
    module load "cuda-toolkit/$CUDA_MODULE_VERSION"   # Protenix needs CUDA_HOME to build its CUDA extensions
    export CUDA_HOME="$CUDA_HOME_PATH"
    abcfold "$ANY_JSON" "$ABCFOLD_OUT_DIR/_prime" \
        $MODEL_FLAG \
        --model_params "$MODEL_PARAMS" \
        $( [[ -n "$AF3_SIF_PATH" ]] && echo "--af3_sif_path $AF3_SIF_PATH" ) \
        --dry_run \
        --override
    echo "Priming complete."
    exit 0
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -d "$FOLD_IN_DIR" ]]; then
    echo "ERROR: $FOLD_IN_DIR not found. Run worflows/preprocessing/Snakefile first."
    exit 1
fi
if [[ -z "$AF3_SIF_PATH" ]] && [[ "$MODEL_FLAG" == *a* ]]; then
    echo "WARNING: AF3_SIF_PATH is empty and AlphaFold3 (-a) is selected —"
    echo "         auto-discovery of IFB's alphafold/$AF3_MODULE_VERSION .sif failed"
    echo "         (see discover_af3_sif() above). ABCfold will fall back to"
    echo "         'docker run', which is normally unavailable on IFB compute"
    echo "         nodes. Set AF3_SIF_PATH in this script (or config.yaml's"
    echo "         abcfold.af3_sif_path) manually, or check that"
    echo "         /shared/software/singularity/wrappers/alphafold/$AF3_MODULE_VERSION/run_alphafold.py"
    echo "         still exists and still references a .sif file."
fi

# ── Collect pending runs, Gibberellin (GA1) importers first ──────────────────
# data/fold_inputs/priority_gibberellin.txt (written by
# worflows/preprocessing/Snakefile's write_priority_manifest rule) lists
# apo/holo run identifiers for the GA1 group. This is the class of protein
# notebook/tm_conformation_clustering_gibberellin_boltz.ipynb showed AF3
# alone under-samples, so it gets the cluster's attention first.
declare -A IS_PRIORITY
if [[ -f "$PRIORITY_MANIFEST" ]]; then
    while IFS= read -r run; do
        [[ -z "$run" ]] && continue
        IS_PRIORITY["$run"]=1
    done < "$PRIORITY_MANIFEST"
fi

PRIORITY_JSONS=()
PRIORITY_DONE=()
REST_JSONS=()
REST_DONE=()
skipped=0

for JSON in "$FOLD_IN_DIR"/*/fold_input.resolved.json; do
    [[ -f "$JSON" ]] || continue
    PROTEIN=$(basename "$(dirname "$JSON")")
    DONE_FILE="$ABCFOLD_OUT_DIR/$PROTEIN/prediction.done"
    if [[ -f "$DONE_FILE" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    if [[ -n "${IS_PRIORITY[$PROTEIN]:-}" ]]; then
        PRIORITY_JSONS+=("$JSON")
        PRIORITY_DONE+=("$DONE_FILE")
    else
        REST_JSONS+=("$JSON")
        REST_DONE+=("$DONE_FILE")
    fi
done

PENDING_JSONS=()
PENDING_DONE=()
if [[ ${#PRIORITY_JSONS[@]} -gt 0 ]]; then
    PENDING_JSONS+=("${PRIORITY_JSONS[@]}")
    PENDING_DONE+=("${PRIORITY_DONE[@]}")
fi
if [[ ${#REST_JSONS[@]} -gt 0 ]]; then
    PENDING_JSONS+=("${REST_JSONS[@]}")
    PENDING_DONE+=("${REST_DONE[@]}")
fi

TOTAL=${#PENDING_JSONS[@]}
N_TASKS=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))   # ceiling division
LAST_TASK=$(( N_TASKS - 1 ))
ARRAY_SPEC="0-${LAST_TASK}%${MAX_CONCURRENT}"

echo "============================================================"
echo " ABCfold SLURM job array submission"
echo " Pending runs (apo+holo) : $TOTAL  (${#PRIORITY_JSONS[@]} Gibberellin-importer, first)"
echo " Already done            : $skipped"
echo " Models                  : $MODEL_FLAG"
echo " Models/recycles per run : $NUMBER_OF_MODELS / $NUM_RECYCLES"
echo " Batch size              : $BATCH_SIZE run(s)/task"
echo " Array tasks              : $N_TASKS  (--array=${ARRAY_SPEC})"
echo " Max concurrent           : $MAX_CONCURRENT"
echo " GRES                     : $GRES"
echo " Time limit/task          : ${TIME} min"
if $DRY_RUN; then
    echo " Mode                     : DRY RUN (nothing submitted)"
fi
if $TEST_MODE; then
    echo " Mode                     : TEST — task 0 only ($(( BATCH_SIZE < TOTAL ? BATCH_SIZE : TOTAL )) run(s))"
fi
echo "============================================================"
echo ""

if [[ $TOTAL -eq 0 ]]; then
    echo "Nothing to do — all predictions already complete."
    exit 0
fi

# ── Write the batch manifest ──────────────────────────────────────────────────
# One line per pending run: "resolved_json|out_dir|done_file". Gibberellin
# (GA1) importers occupy the first lines, so lower array-task indices (and
# therefore the earliest --max-concurrent slots) process them first.

MANIFEST_DIR="$ABCFOLD_OUT_DIR/array_manifest"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/manifest.txt"

: > "$MANIFEST"   # truncate
for (( i=0; i<TOTAL; i++ )); do
    json="${PENDING_JSONS[$i]}"
    done_file="${PENDING_DONE[$i]}"
    protein=$(basename "$(dirname "$json")")
    out_dir="$ABCFOLD_OUT_DIR/$protein"
    mkdir -p "$(dirname "$out_dir")"
    echo "${json}|${out_dir}|${done_file}" >> "$MANIFEST"
done

echo "Manifest written: $MANIFEST ($TOTAL lines, Gibberellin importers first)"
echo ""

if $DRY_RUN; then
    echo "Dry-run — array tasks breakdown:"
    for (( task=0; task<N_TASKS; task++ )); do
        start=$(( task * BATCH_SIZE ))
        end=$(( start + BATCH_SIZE - 1 ))
        [[ $end -ge $TOTAL ]] && end=$(( TOTAL - 1 ))
        count=$(( end - start + 1 ))
        echo "  Task $task: $count run(s) (lines $((start+1))-$((end+1)))"
        for (( i=start; i<=end; i++ )); do
            json="${PENDING_JSONS[$i]}"
            protein=$(basename "$(dirname "$json")")
            marker=""
            [[ -n "${IS_PRIORITY[$protein]:-}" ]] && marker=" [GA1 priority]"
            echo "    $protein$marker"
        done
    done
    echo ""
    echo "Would submit: sbatch --array=${ARRAY_SPEC} ..."
    exit 0
fi

# ── Submit the job array ──────────────────────────────────────────────────────
LOG_DIR="$MANIFEST_DIR/logs"
mkdir -p "$LOG_DIR"

if $TEST_MODE; then
    ARRAY_SPEC="0"
    echo "TEST MODE: submitting task 0 only (${BATCH_SIZE} run(s))"
    echo "           Once it completes successfully, run without --test for the full array."
    echo ""
fi

ACCOUNT_FLAG=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_FLAG="--account=$ACCOUNT"

sbatch \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --gres="$GRES" \
    --time="$TIME" \
    --array="${ARRAY_SPEC}" \
    $ACCOUNT_FLAG \
    --job-name="abcfold_array" \
    --output="$LOG_DIR/task_%a.log" \
    --error="$LOG_DIR/task_%a.err" \
    << SLURM_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

TASK_ID=\$SLURM_ARRAY_TASK_ID
BATCH_SIZE=$BATCH_SIZE
MANIFEST="$MANIFEST"
MODEL_FLAG="$MODEL_FLAG"
MODEL_PARAMS="$MODEL_PARAMS"
AF3_SIF_PATH="$AF3_SIF_PATH"
CUDA_HOME_PATH="$CUDA_HOME_PATH"
CUDA_MODULE_VERSION="$CUDA_MODULE_VERSION"
NUMBER_OF_MODELS=$NUMBER_OF_MODELS
NUM_RECYCLES=$NUM_RECYCLES

module load singularity   # ABCfold shells out to \`singularity exec\` directly for AF3
                          # (not \`module load alphafold\` — we only borrow its .sif, see
                          # discover_af3_sif() at submission time)
module load "cuda-toolkit/\$CUDA_MODULE_VERSION"   # Protenix needs CUDA_HOME to build its CUDA extensions
export CUDA_HOME="\$CUDA_HOME_PATH"

echo "[\$(date)] Array task \$TASK_ID — batch size $BATCH_SIZE"
echo "  Manifest: \$MANIFEST"
echo "  abcfold: \$(which abcfold)"
echo "  singularity: \$(which singularity)"

# Compute line range for this task (1-based for sed)
LINE_START=\$(( TASK_ID * BATCH_SIZE + 1 ))
LINE_END=\$(( LINE_START + BATCH_SIZE - 1 ))

echo "[\$(date)] Processing manifest lines \$LINE_START-\$LINE_END"
echo ""

while IFS='|' read -r json out_dir done_file; do
    [[ -z "\$json" ]] && continue

    protein=\$(basename "\$(dirname "\$json")")

    if [[ -f "\$done_file" ]]; then
        echo "[\$(date)] SKIP: \$protein (already done)"
        continue
    fi

    echo "[\$(date)] START abcfold \$MODEL_FLAG: \$protein"
    abcfold "\$json" "\$out_dir" \\
        \$MODEL_FLAG \\
        --model_params "\$MODEL_PARAMS" \\
        \$( [[ -n "\$AF3_SIF_PATH" ]] && echo "--af3_sif_path \$AF3_SIF_PATH" ) \\
        --number_of_models "\$NUMBER_OF_MODELS" \\
        --num_recycles "\$NUM_RECYCLES" \\
        --no_server \\
        --no_visuals \\
        --override

    echo "\$(date): prediction finished" > "\$done_file"
    echo "[\$(date)] DONE: \$protein"
    echo ""

done < <(sed -n "\${LINE_START},\${LINE_END}p" "\$MANIFEST")

echo "[\$(date)] Array task \$TASK_ID complete."
SLURM_SCRIPT

echo "Job array submitted: --array=${ARRAY_SPEC}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/task_0.log"
