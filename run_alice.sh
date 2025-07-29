#!/usr/bin/env bash

ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/results"

mkdir -p ${RESULTS_ROOT}

### NAME YOUR EXPERIMENT HERE ##
EXP_NAME="gbert+a+q"
################################

## Local variables for current experiment
EXP_ROOT="${RESULTS_ROOT}/${EXP_NAME}"
# export HF_HOME="/home/hf_home"
export WANDB_PROJECT="alice-rubrics"
export WANDB_NAME="${EXP_NAME}"
export WANDB_NOTES=""
export WANDB_TAGS="gbert,multiseq"
mkdir -p ${EXP_ROOT}
#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python src/train_rubretrieval.py --save-dir "${EXP_ROOT}" \
    --base-model "bert-base-uncased" \
    --batch-size 4 \
    --train-frac 0.01 \
    --lr 3e-5 \
    --max-epoch 5

            
                
               
