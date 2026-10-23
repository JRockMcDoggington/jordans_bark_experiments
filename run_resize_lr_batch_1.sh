#!/bin/bash -l
#
#SBATCH --job-name=resize_lr_batch_1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

trap 'status=$?; echo "Batch 1 finished: $(date) (exit status ${status})"' EXIT

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Error: run this script with sbatch." >&2
  exit 1
fi

EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SLURM_SUBMIT_DIR}"
TRAIN_SCRIPT="${EXPERIMENT_DIR}/training_script.py"
MANIFEST_PATH="${EXPERIMENT_DIR}/resize_lr_experiment_manifest.tsv"
RESULTS_DIR="${EXPERIMENT_DIR}/experiment_results"
CHECKPOINTS_DIR="${EXPERIMENT_DIR}/checkpoints"
CSV_PATH="${EXPERIMENT_DIR}/image_dir/labels.csv"
PHOTOS_PATH="${EXPERIMENT_DIR}/image_dir/images"
configuration_indices=(0 16 24 26 32 35 40 48 56 64 72 88 96)

for required_file in "${TRAIN_SCRIPT}" "${MANIFEST_PATH}" "${CSV_PATH}"; do
  if [[ ! -f "${required_file}" || ! -s "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${PHOTOS_PATH}" ]]; then
  echo "Required photo directory does not exist: ${PHOTOS_PATH}" >&2
  exit 1
fi

cd "${EXPERIMENT_DIR}"
mkdir -p "${RESULTS_DIR}" "${CHECKPOINTS_DIR}" torch_cache

header="$(head -n 1 "${MANIFEST_PATH}")"
header="${header%$'\r'}"
if [[ "${header}" != $'experiment_id\tmodel\tlearning_rate\tresize_strategy\tcrop_size_pixels' ]]; then
  echo "Malformed manifest header in ${MANIFEST_PATH}." >&2
  exit 1
fi
manifest_rows="$(awk 'END { print NR - 1 }' "${MANIFEST_PATH}")"
if [[ "${manifest_rows}" != "99" ]]; then
  echo "Manifest must contain exactly 99 data rows; found ${manifest_rows}." >&2
  exit 1
fi

source "${EXPERIMENT_DIR}/load_modules.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${EXPERIMENT_DIR}/torch_cache"

echo "resize/LR experiment batch 1 of 8"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Configuration count: ${#configuration_indices[@]}"
echo "Zero-based manifest indices: ${configuration_indices[*]}"
echo "Started: $(date)"

failed_configurations=()
for configuration_index in "${configuration_indices[@]}"; do
  manifest_line="$((configuration_index + 2))"
  mapfile -t fields < <(
    awk -F '\t' -v target="${manifest_line}" '
      NR == target {
        sub(/\r$/, "", $0)
        for (i = 1; i <= 5; i++) print $i
        print NF
        found = 1
      }
      END { if (!found) exit 2 }
    ' "${MANIFEST_PATH}"
  )
  if [[ "${#fields[@]}" -ne 6 || "${fields[5]:-}" != "5" ]]; then
    echo "Could not parse five fields from manifest index ${configuration_index}." >&2
    exit 1
  fi

  experiment_id="${fields[0]}"
  model="${fields[1]}"
  learning_rate="${fields[2]}"
  resize_strategy="${fields[3]}"
  crop_size_pixels="${fields[4]}"

  case "${model}" in
    efficientnet_b0|efficientnet_b1|efficientnet_b2|efficientnet_b3|efficientnet_b4|efficientnet_b5|efficientnet_b6|efficientnet_b7|efficientnet_v2_s|efficientnet_v2_m|efficientnet_v2_l) ;;
    *) echo "Unsupported model at manifest index ${configuration_index}: ${model}" >&2; exit 1 ;;
  esac
  case "${learning_rate}" in
    0.001|0.0001|0.0003) ;;
    *) echo "Invalid learning rate at manifest index ${configuration_index}: ${learning_rate}" >&2; exit 1 ;;
  esac
  case "${resize_strategy}:${crop_size_pixels}" in
    resize:|big_center_crop_resize:960|small_center_crop_resize:480) ;;
    *) echo "Invalid resize/crop combination at manifest index ${configuration_index}: ${resize_strategy}:${crop_size_pixels}" >&2; exit 1 ;;
  esac

  echo
  echo "===== Starting ${experiment_id} (manifest index ${configuration_index}) at $(date) ====="
  if python "${TRAIN_SCRIPT}" \
    --csv-path "${CSV_PATH}" \
    --photos-path "${PHOTOS_PATH}" \
    --experiment-id "${experiment_id}" \
    --model "${model}" \
    --learning-rate "${learning_rate}" \
    --resize-strategy "${resize_strategy}" \
    --auto-batch-fallback \
    --num-workers 4 \
    --require-cached-imagenet-weights; then
    echo "===== ${experiment_id} completed successfully ====="
  else
    exit_status=$?
    echo "===== ${experiment_id} failed with exit status ${exit_status}; continuing =====" >&2
    failed_configurations+=("${configuration_index}:${experiment_id}:${exit_status}")
  fi
done

if (( ${#failed_configurations[@]} > 0 )); then
  echo "Batch 1 failed configurations (index:experiment:exit):" >&2
  printf '  %s\n' "${failed_configurations[@]}" >&2
  exit 1
fi

echo "All batch 1 configurations completed successfully."
