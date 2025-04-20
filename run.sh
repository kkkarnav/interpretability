#!/bin/bash

source /apps/miniconda3/bin/activate interpret
python ./interpret/dictlearning_act.py $ARGS
conda deactivate

