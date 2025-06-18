#!/usr/bin/env bash

ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/results"

mkdir -p ${RESULTS_ROOT}

### NAME YOUR EXPERIMENT HERE ##
EXP_NAME="grasp-bert"
################################

## Local variables for current experiment
EXP_ROOT="${RESULTS_ROOT}/${EXP_NAME}"
export WANDB_PROJECT="alice-asag"
export WANDB_NAME="${EXP_NAME}"
mkdir -p ${EXP_ROOT}
#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python src/train_grasp.py --save-dir "${EXP_ROOT}" \
    --model-name "deepset/gbert-large" \
    --no-save \
    --batch-size 16 \
    --lr 2e-5 \
    --lr2 5e-4 \
    --max-epoch 5 \
    --freeze-layers 6  \
    --freeze-embeddings
            
                
               
