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
export WANDB_NAME="${EXP_NAME}"
export WANDB_NOTES=""
mkdir -p ${EXP_ROOT}
#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
accelerate launch src/train_xnet.py --save-dir "${EXP_ROOT}" \
    --base-model "bert-base-uncased" \
    --batch-size 16 \
    --train-frac 1 \
    --lr 3e-5 \
    --max-epoch 5 \
    --use-lora 2>&1 | tee "out.log"

            
                
               
