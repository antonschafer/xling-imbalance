#!/bin/bash

# adapted from https://github.com/babylm/evaluation-pipeline/blob/17b3806376fa90c24653efe0c6c571adfb6a72c7/finetune_all_tasks.sh

MODEL_PATH=$1
LR=${2:-5e-5}
PATIENCE=${3:-10}
BSZ=${4:-64}
EVAL_EVERY=${5:-200}
MAX_EPOCHS=${6:-10}
SEED=${7:-12}

# Fine-tune and evaluate on (Super)GLUE tasks
# If your system uses sbatch or qsub, consider using that to parallelize calls to finetune_model.sh
for subtask in {"cola","sst2","mrpc","qqp","mnli","mnli-mm","qnli","rte","boolq","multirc","wsc"}; do
    ./finetune_task.sh $MODEL_PATH glue $subtask $LR $PATIENCE $BSZ $EVAL_EVERY $MAX_EPOCHS $SEED
done


