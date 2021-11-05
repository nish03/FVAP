#!/usr/bin/sh

JOBS_DIR=$(dirname $0)

rm -rf ${JOBS_DIR}/experiments
find config -iname "*.args" | xargs python3 ${JOBS_DIR}/generate_experiment_jobs.py
