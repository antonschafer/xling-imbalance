#!/bin/bash

# adapted from https://github.com/babylm/evaluation-pipeline/blob/17b3806376fa90c24653efe0c6c571adfb6a72c7/finetune_model.sh

WANDB_RUN=$1
TASK_NAME=$2
SUBTASK_NAME=$3
LR=${4:-5e-5}           # default: 5e-5
PATIENCE=${5:-10}       # default: 10
BSZ=${6:-64}            # default: 64
EVAL_EVERY=${7:-200}    # default: 200
MAX_EPOCHS=${8:-10}     # default: 10
SEED=${9:-12}           # default: 12

if [[ "$SUBTASK_NAME" = "mnli" ]]; then
    VALID_NAME="validation_matched"
    OUT_DIR="mnli"
elif [[ "$SUBTASK_NAME" = "mnli-mm" ]]; then
    VALID_NAME="validation_mismatched"
    SUBTASK_NAME="mnli"
    OUT_DIR="mnli-mm"
else
    VALID_NAME="validation"
    OUT_DIR=$SUBTASK_NAME
fi

echo $WANDB_RUN
echo $TASK_NAME
echo $SUBTASK_NAME

TASK_DATA_DIR=$(pwd)/data/babylm/filter-data/${TASK_NAME}_filtered


export WANDB_DISABLED=true

ACC_STEPS=2
# # Compute acc steps based on GPU memory
# gpu_mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
# gpu_mem_gb=$(( gpu_mem_mb / 1024 )) # convert to GB
# if (( gpu_mem_gb > 20 )); then
#     ACC_STEPS=2
# else
#     ACC_STEPS=4
# fi

venv/bin/python -m languini.projects.gpt.finetuning.finetune_classification \
  --wandb_run $WANDB_RUN \
  --output_dir finetune_results/${WANDB_RUN//\//_}/$OUT_DIR/ \
  --train_file $TASK_DATA_DIR/$SUBTASK_NAME.train.json \
  --validation_file $TASK_DATA_DIR/$SUBTASK_NAME.$VALID_NAME.json \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $(($BSZ/$ACC_STEPS)) \
  --learning_rate $LR \
  --num_train_epochs $MAX_EPOCHS \
  --evaluation_strategy steps \
  --patience $PATIENCE \
  --eval_every $EVAL_EVERY \
  --eval_steps $EVAL_EVERY \
  --save_strategy no \
  --overwrite_output_dir \
  --overwrite_cache true \
  --gradient_accumulation_steps $ACC_STEPS \
  --seed $SEED \
  --L_train L1 \
  --use_cpu true \


