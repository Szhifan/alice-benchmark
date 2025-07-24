#!/usr/bin/env bash

ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/results"

mkdir -p ${RESULTS_ROOT}

### NAME YOUR EXPERIMENT HERE ##
EXP_NAME="mbert"
################################

## Local variables for current experiment
EXP_ROOT="${RESULTS_ROOT}/${EXP_NAME}"
export HF_HOME="/home/hf_home"
export WANDB_PROJECT="alice-rubrics"
export WANDB_NAME="${EXP_NAME}"
export WANDB_NOTES="Training Alice with mBERT base model"
export WANDB_TAGS="mbert,alice,rubrics"
mkdir -p ${EXP_ROOT}
#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python src/train_seq.py --save-dir "${EXP_ROOT}" \
    --base-model "google-t5/t5-base" \
    --batch-size 16 \
    --train-frac 1 \
    --lr 2e-5 \
    --lr2 5e-4 \
    --max-epoch 5

            
                
               
