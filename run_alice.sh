#!/usr/bin/env bash
EXP_NAME="llama3.2-1b-ke"
ROOT=$(git rev-parse --show-toplevel)
## Local variables for current experiment
EXP_ROOT="${ROOT}/results/${EXP_NAME}"
# export HF_HOME="/home/hf_home"
export WANDB_NAME="${EXP_NAME}"
export WANDB_NOTES=""
echo "Experiment root: ${EXP_ROOT}"
mkdir -p ${EXP_ROOT}
#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
accelerate launch src/train_xnet_llm.py --save-dir "${EXP_ROOT}" \
    --base-model "meta-llama/Llama-3.2-1b" \
    --batch-size 4 \
    --train-frac 1 \
    --gradient-accumulation-steps 8 \
    --lr 1e-4 \
    --max-epoch 5 \
    --use-bnb  \
    --task-name "ke" \
    --bf16 \
    --use-lora 2>&1 | tee "out.log"

            
                
               
