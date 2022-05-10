#!/bin/bash

EXPERIMENT_DIR=$1
EXPERIMENT_ARGS_FILES="$(find $EXPERIMENT_DIR -iname "*.args" -type f -print0 | xargs -0 realpath --relative-to=$EXPERIMENT_DIR)"
START_EXPERIMENT_SCRIPT="$(dirname $0)/start_experiment_job.sh"
for EXPERIMENT_ARGS_FILE in $EXPERIMENT_ARGS_FILES; do
  echo "Launching $START_EXPERIMENT_SCRIPT $EXPERIMENT_DIR $EXPERIMENT_ARGS_FILE"
  $START_EXPERIMENT_SCRIPT $EXPERIMENT_DIR $EXPERIMENT_ARGS_FILE
done

