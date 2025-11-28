#!/usr/bin/env bash
EXP_NAME="mbert"
ROOT=$(git rev-parse --show-toplevel)
## Local variables for current experiment
EXP_ROOT="${ROOT}/results/${EXP_NAME}"
# export HF_HOME="/home/hf_home"
export WANDB_NAME="${EXP_NAME}"
export WANDB_NOTES=""
echo "Experiment root: ${EXP_ROOT}"
mkdir -p ${EXP_ROOT}
#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python src/train.py --save-dir "${EXP_ROOT}" \
    --base-model "bert-base-multilingual-cased" \
    --batch-size 4 \
    --train-frac 0.01 \
    --gradient-accumulation-steps 8 \
    --lr 1e-4 \
    --max-epoch 5 \
    --lora-rank 8 \
    --use-lora 2>&1 | tee "out.log"

            
                
               
