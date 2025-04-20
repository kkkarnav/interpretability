#!/bin/bash

source /apps/miniconda3/bin/activate tai

# Define the common variables
QUEUE="cpu"
OUTPUT_DIR="./out/sae2"
ERROR_DIR="./err/sae2"
ARGS_BASE=""
SCRIPT="./scripts/run.sh"

JOB_NAME="gpt_eval"
ARGS="$ARGS_BASE"
qsub -q $QUEUE -N "$JOB_NAME" -o $OUTPUT_DIR -e $ERROR_DIR -l host=compute4 -v "ARGS='$ARGS'" $SCRIPT

conda deactivate

