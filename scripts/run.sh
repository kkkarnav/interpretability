#!/bin/bash

source /apps/miniconda3/bin/activate tai
python ./interpret/scripts/sae_train.py $ARGS
conda deactivate

