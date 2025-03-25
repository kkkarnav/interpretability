#!/bin/bash

source /apps/miniconda3/bin/activate interpret

# Define the common variables
QUEUE="gpu"
OUTPUT_DIR="out"
ERROR_DIR="err"
ARGS_BASE=""
SCRIPT="./run.sh"

JOB_NAME="dictlearning"
ARGS="$ARGS_BASE"
qsub -q $QUEUE -N "$JOB_NAME" -o $OUTPUT_DIR -e $ERROR_DIR -v "ARGS='$ARGS'" $SCRIPT

conda deactivate

