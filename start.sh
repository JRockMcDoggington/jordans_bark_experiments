#!/bin/bash -l
# resize/LR experiment entry point. Invoke with bash from a Sapelo2 login node.
#
#   bash start.sh                  # same as --prepare: queue full preparation
#   bash start.sh --prepare         # always queue full weight preparation
#   bash start.sh --run             # prepare missing weights, submit eight jobs
#   bash start.sh --prepare --run    # always prepare before the eight jobs
#   bash start.sh --reset           # rebuild run scripts, download fresh weights
#   bash start.sh --reset --run      # reset preparation, then submit eight jobs
#
# Default project: the directory containing this start.sh file.
# Optional environment overrides: EXPERIMENT_DIR and EXPERIMENT_RUN_SCRIPTS_DIR.
# Run scripts are selected from generated_runs/ when present, otherwise the
# project directory. Missing/incomplete sets are generated from the packaged
# resize_lr_experiment_manifest.tsv and run_resize_lr_batch_1.sh in the project directory.
# Package generate_run_scripts.sh with these files. The original manifest and
# batch-1 template are preserved; changed generated copies are backed up.
# Each invocation of --run submits a new set of jobs.
# Every --run queues a CPU dataset preflight. It also prepares weights if an
# expected weight file is missing, empty, or unreadable. Training waits for success.
# Lightweight checks test file presence/readability/size, not checkpoint contents.
# Reset regenerates all eight run scripts from the packaged source template.
# The preparation job removes only this project's torch_cache before caching
# weights again. Source files, folds, datasets, logs, results, and trained
# checkpoints are preserved. Reset requires no active jobs in the project.

set -euo pipefail

error() {
  echo "Error: $*" >&2
  exit 1
}

usage() {
  echo "Usage: bash start.sh [--prepare] [--run] [--reset]"
  echo "Default project: the directory containing start.sh."
  echo "All modes generate a complete set of run scripts when any are missing."
  echo "Default/--prepare: always submit a CPU job to verify/cache all eleven weights."
  echo "--run: queue CPU dataset validation/preparation, then eight dependent GPU jobs."
  echo "--prepare --run: always prepare first; training waits for successful preparation."
  echo "--reset: regenerate all run scripts and download fresh weights."
  echo "--reset implies --prepare; add --run to queue training after preparation succeeds."
  echo "Reset preserves source files, folds, datasets, logs, results, and trained checkpoints."
  echo "--run and --reset require no active jobs in this project directory."
}

# DEFAULT filenames for the torchvision/0.16.2 module used by the run scripts.
# Source: https://docs.pytorch.org/vision/0.16/_modules/torchvision/models/efficientnet.html
# B1 uses IMAGENET1K_V2; the other models use IMAGENET1K_V1.
weight_filenames=(
  efficientnet_b0_rwightman-7f5810bc.pth
  efficientnet_b1-c27df63c.pth
  efficientnet_b2_rwightman-c35c1473.pth
  efficientnet_b3_rwightman-b3899882.pth
  efficientnet_b4_rwightman-23ab8bcd.pth
  efficientnet_b5_lukemelas-1a07897c.pth
  efficientnet_b6_lukemelas-24a108a5.pth
  efficientnet_b7_lukemelas-c5b4e57e.pth
  efficientnet_v2_s-dd5fe13b.pth
  efficientnet_v2_m-dc08266a.pth
  efficientnet_v2_l-59c71312.pth
)

missing_weights=()
check_cached_weights() {
  missing_weights=()
  local filename path
  for filename in "${weight_filenames[@]}"; do
    path="${EXPERIMENT_DIR}/torch_cache/hub/checkpoints/${filename}"
    if [[ ! -f "${path}" || ! -r "${path}" || ! -s "${path}" ]]; then
      missing_weights+=("${path}")
    fi
  done
  (( ${#missing_weights[@]} == 0 ))
}

require_file() {
  [[ -f "$1" && -r "$1" && -s "$1" ]] || error "Required file is missing, empty, or unreadable: $1"
}

require_idle_project() {
  command -v squeue >/dev/null 2>&1 || error "squeue is unavailable; submission must run on Sapelo2."
  local queue_output job_id job_name job_directory
  local active_jobs=()
  if ! queue_output="$(squeue --noheader --user="$(id -un)" --format='%i|%j|%Z')"; then
    error "Could not check active jobs; no new jobs were submitted."
  fi
  while IFS='|' read -r job_id job_name job_directory; do
    [[ -n "${job_id}" ]] || continue
    if [[ -d "${job_directory}" ]]; then
      job_directory="$(cd -- "${job_directory}" && pwd -P)"
    fi
    if [[ "${job_directory}" == "${EXPERIMENT_DIR}" ]]; then
      active_jobs+=("${job_id} (${job_name})")
    fi
  done <<< "${queue_output}"
  if (( ${#active_jobs[@]} > 0 )); then
    error "This project's queued/running jobs must finish or be cancelled before another submission: ${active_jobs[*]}"
  fi
}

ensure_run_scripts() {
  local batch run_script generation_dir generator_script
  local generation_needed="${reset_requested}"
  if [[ -d "${run_scripts_dir}" ]]; then
    run_scripts_dir="$(cd -- "${run_scripts_dir}" && pwd -P)"
  fi
  for batch in {1..8}; do
    run_script="${run_scripts_dir}/run_resize_lr_batch_${batch}.sh"
    if [[ ! -f "${run_script}" || ! -r "${run_script}" || ! -s "${run_script}" ]]; then
      generation_needed=true
      break
    fi
  done
  if [[ "${generation_needed}" == false ]]; then
    echo "All eight run scripts are already present: ${run_scripts_dir}"
    return
  fi

  # Batch 1 is the packaged source template. Generate copies in a separate
  # directory so the original batch 1 cannot be replaced.
  require_file "${EXPERIMENT_DIR}/resize_lr_experiment_manifest.tsv"
  require_file "${EXPERIMENT_DIR}/run_resize_lr_batch_1.sh"
  generator_script="${EXPERIMENT_DIR}/generate_run_scripts.sh"
  if [[ ! -f "${generator_script}" ]]; then
    generator_script="$(dirname -- "${script_path}")/generate_run_scripts.sh"
  fi
  require_file "${generator_script}"
  if [[ "${run_scripts_dir}" == "${EXPERIMENT_DIR}" ]]; then
    generation_dir="${EXPERIMENT_DIR}/generated_runs"
  else
    generation_dir="${run_scripts_dir}"
  fi

  if [[ "${reset_requested}" == true ]]; then
    echo "Preparation reset requested; regenerating all eight run scripts."
  else
    echo "Run scripts are missing or incomplete; generating all eight."
  fi
  bash "${generator_script}" \
    "${EXPERIMENT_DIR}/resize_lr_experiment_manifest.tsv" \
    "${EXPERIMENT_DIR}/run_resize_lr_batch_1.sh" \
    8 "${generation_dir}" || error "Run-script generation failed; no jobs were submitted."
  run_scripts_dir="$(cd -- "${generation_dir}" && pwd -P)"
  for batch in {1..8}; do
    require_file "${run_scripts_dir}/run_resize_lr_batch_${batch}.sh"
  done
}

prepare_project() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || error "Weight preparation requires a Slurm compute allocation."
  [[ -d "${EXPERIMENT_DIR}" ]] || error "Project directory does not exist: ${EXPERIMENT_DIR}"
  cd -- "${EXPERIMENT_DIR}"
  EXPERIMENT_DIR="$(pwd -P)"
  export EXPERIMENT_DIR
  [[ -f training_script.py ]] || error "Missing ${EXPERIMENT_DIR}/training_script.py"
  mkdir -p logs

  source "${EXPERIMENT_DIR}/load_modules.sh"

  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
  export PYTHONUNBUFFERED=1
  export TORCH_HOME="${EXPERIMENT_DIR}/torch_cache"
  if [[ "${check_data_requested}" == true ]]; then
    echo "Validating the dataset for every resize strategy before GPU training."
    python training_script.py --validate-data-only --resize-strategy big_center_crop_resize --seed 0
  fi
  if [[ "${reset_requested}" == true ]]; then
    echo "Resetting preparation cache: ${TORCH_HOME}"
    # No trailing slash: a torch_cache symlink is removed without following it.
    rm -rf -- "${TORCH_HOME}"
    echo "All eleven pretrained weights will be downloaded and verified again."
  fi
  mkdir -p -- "${TORCH_HOME}"
  if [[ "${skip_weight_verification}" == false ]]; then
    echo "Preparing pretrained weights in ${TORCH_HOME}"
    python training_script.py --verify-imagenet-weights-only
  fi
  echo "Experiment preparation completed successfully."
}

# Private entry point for the CPU job submitted below. Pass the project path
# explicitly because Slurm executes a copy of this file from its spool directory.
if [[ "${1:-}" == "--prepare-job" ]]; then
  (( $# >= 2 )) || error "Internal preparation requires the project directory."
  export EXPERIMENT_DIR="$2"
  shift 2
  reset_requested=false
  check_data_requested=false
  skip_weight_verification=false
  for option in "$@"; do
    case "${option}" in
      --reset) reset_requested=true ;;
      --check-data) check_data_requested=true ;;
      --skip-weight-verification) skip_weight_verification=true ;;
      *) error "Unknown internal preparation option: ${option}" ;;
    esac
  done
  prepare_project
  exit 0
fi

prepare_requested=false
run_requested=false
reset_requested=false
(( $# > 0 )) || prepare_requested=true
for argument in "$@"; do
  case "${argument}" in
    --prepare) prepare_requested=true ;;
    --run) run_requested=true ;;
    --reset) reset_requested=true; prepare_requested=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; error "Unknown argument: ${argument}" ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${script_dir}}"
[[ -d "${EXPERIMENT_DIR}" ]] || error "Project directory does not exist: ${EXPERIMENT_DIR}"
EXPERIMENT_DIR="$(cd -- "${EXPERIMENT_DIR}" && pwd -P)"
export EXPERIMENT_DIR
require_file "${EXPERIMENT_DIR}/training_script.py"
require_file "${EXPERIMENT_DIR}/load_modules.sh"
script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
logs_dir="${EXPERIMENT_DIR}/logs"

if [[ -n "${EXPERIMENT_RUN_SCRIPTS_DIR:-}" ]]; then
  run_scripts_dir="${EXPERIMENT_RUN_SCRIPTS_DIR}"
elif [[ -d "${EXPERIMENT_DIR}/generated_runs" ]]; then
  run_scripts_dir="${EXPERIMENT_DIR}/generated_runs"
else
  run_scripts_dir="${EXPERIMENT_DIR}"
fi

if [[ "${reset_requested}" == true || "${run_requested}" == true ]]; then
  command -v sbatch >/dev/null 2>&1 || error "sbatch is unavailable; run this on Sapelo2."
  require_idle_project
fi

# Both preparation and running can start from a package containing only batch 1.
# This generates text files locally and does not submit any job itself.
ensure_run_scripts

# Check all inputs needed for submission before queueing any job.
if [[ "${run_requested}" == true ]]; then
  for filename in resize_lr_experiment_manifest.tsv; do
    require_file "${EXPERIMENT_DIR}/${filename}"
  done

  # Keep these dataset paths in agreement with the paths in your run scripts.
  data_folder="${EXPERIMENT_DIR}/image_dir"
  require_file "${data_folder}/labels.csv"
  photos_dir="${data_folder}/images"
  [[ -d "${photos_dir}" && -r "${photos_dir}" && -x "${photos_dir}" ]] || error "Photo directory is missing or inaccessible: ${photos_dir}"
  [[ -n "$(find "${photos_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || error "Photo directory is empty: ${photos_dir}"

  [[ -d "${run_scripts_dir}" ]] || error "Run-script directory does not exist: ${run_scripts_dir}"
  run_scripts_dir="$(cd -- "${run_scripts_dir}" && pwd -P)"
  for batch in {1..8}; do
    run_script="${run_scripts_dir}/run_resize_lr_batch_${batch}.sh"
    require_file "${run_script}"
    bash -n "${run_script}" || error "Invalid Bash syntax: ${run_script}"
  done
fi

# Logs must exist before Slurm starts either preparation or training.
# Use the selected project's logs directory for every submitted job.
mkdir -p "${logs_dir}"
if [[ "${reset_requested}" == false ]]; then
  mkdir -p "${EXPERIMENT_DIR}/torch_cache"
fi

# Every run uses CPU preflight so invalid labels/images fail before GPU work.
# Existing caches avoid downloading/loading all eleven models unnecessarily.
skip_weight_verification=false
if [[ "${prepare_requested}" == false ]]; then
  if check_cached_weights; then
    echo "All eleven expected weight files are present, readable, and nonempty."
    echo "Queueing dataset preflight; cached weights will be loaded by training."
    skip_weight_verification=true
  else
    echo "Cached weights need preparation (${#missing_weights[@]} missing, empty, or unreadable files)."
  fi
  prepare_requested=true
fi

prep_job_id=""
if [[ "${prepare_requested}" == true ]]; then
  echo "Submitting CPU preparation/preflight."
  command -v sbatch >/dev/null 2>&1 || error "sbatch is unavailable; run this on Sapelo2."
  prepare_job_options=()
  if [[ "${reset_requested}" == true ]]; then
    prepare_job_options+=(--reset)
  fi
  if [[ "${run_requested}" == true ]]; then
    prepare_job_options+=(--check-data)
  fi
  if [[ "${skip_weight_verification}" == true ]]; then
    prepare_job_options+=(--skip-weight-verification)
  fi
  # CPU-only preparation. Adjust these resources after measuring the real job.
  prep_submission="$(sbatch --parsable \
    --job-name=resize_lr_prepare \
    --partition=batch \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=01:00:00 \
    --chdir="${EXPERIMENT_DIR}" \
    --output="${logs_dir}/prepare_%j.out" \
    --error="${logs_dir}/prepare_%j.err" \
    --export=ALL \
    "${script_path}" --prepare-job "${EXPERIMENT_DIR}" "${prepare_job_options[@]}")"
  prep_job_id="${prep_submission%%;*}"
  [[ "${prep_job_id}" =~ ^[0-9]+$ ]] || error "Unexpected preparation submission response: ${prep_submission}"
  echo "Submitted CPU preparation job: ${prep_job_id}"
  echo "Preparation logs: ${logs_dir}/prepare_${prep_job_id}.out and .err"
fi

if [[ "${run_requested}" == false ]]; then
  if [[ -n "${prep_job_id}" ]]; then
    echo "Preparation is queued; it has not necessarily completed. Check with sq --me."
  fi
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || error "sbatch is unavailable; run this on Sapelo2."
dependency_options=()
if [[ -n "${prep_job_id}" ]]; then
  dependency_options=("--dependency=afterok:${prep_job_id}")
fi

submit_batch() {
  local batch="$1" submission job_id
  if ! submission="$(sbatch --parsable \
    "${dependency_options[@]}" \
    --chdir="${EXPERIMENT_DIR}" \
    --output="${logs_dir}/%x_%j.out" \
    --error="${logs_dir}/%x_%j.err" \
    --export=ALL \
    "${run_scripts_dir}/run_resize_lr_batch_${batch}.sh")"; then
    error "Batch ${batch} submission failed. Previously printed job IDs remain submitted."
  fi
  job_id="${submission%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || error "Unexpected batch ${batch} submission response: ${submission}"
  if [[ -n "${prep_job_id}" ]]; then
    echo "Submitted training batch ${batch}: ${job_id} (waits for preparation ${prep_job_id})"
  else
    echo "Submitted training batch ${batch}: ${job_id}"
  fi
}

# A finite set of eight submissions, with no scheduler polling or retry loop.
echo "Run scripts: ${run_scripts_dir}"
submit_batch 1
submit_batch 2
submit_batch 3
submit_batch 4
submit_batch 5
submit_batch 6
submit_batch 7
submit_batch 8
if [[ -n "${prep_job_id}" ]]; then
  echo "All eight training jobs are queued and can start only after preparation succeeds."
else
  echo "All eight training jobs are submitted. No preparation job was needed."
fi
echo "Check preparation and training status with sq --me."
