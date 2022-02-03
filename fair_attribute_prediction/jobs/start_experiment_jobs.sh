#!/bin/bash

shopt -s nullglob

JOB_DIR=$(dirname "${0}")
PROJECT_DIR=$(realpath "${JOB_DIR}/..")
EXPERIMENT_SLURM_FILE="${JOB_DIR}"/experiment.slurm
for EXPERIMENT_ARGS_DIR in "${@}"; do
  EXPERIMENT_NAME=$(basename "${EXPERIMENT_ARGS_DIR}")
  EXPERIMENT_ARGS_FILES=("$EXPERIMENT_ARGS_DIR"/*.args)
  EXPERIMENT_TASK_COUNT=${#EXPERIMENT_ARGS_FILES[@]}
  EXPERIMENT_TASK_COUNT=$((EXPERIMENT_TASK_COUNT - 1))

  sbatch \
    --array 0-"${EXPERIMENT_TASK_COUNT}" \
    --job-name="${EXPERIMENT_NAME}" \
    --chdir="${PROJECT_DIR}" \
    "${EXPERIMENT_SLURM_FILE}"
done