#!/bin/bash

JOB_FILE=$1
JOB_RUN_COUNT=$2
for JOB_RUN in {1..$JOB_RUN_COUNT}; do
    JOB_CMD="sbatch "
    if [ -n "$DEPENDENCY" ]; then
        JOB_CMD="$JOB_CMD --dependency afterany:$DEPENDENCY"
    fi
    JOB_CMD="$JOB_CMD $JOB_FILE"
    echo -n "Starting job run $JOB_RUN: $JOB_CMD"
    DEPENDENCY=$(echo $($JOB_CMD) | awk '{print $4}')
done
