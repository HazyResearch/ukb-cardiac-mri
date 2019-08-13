#!/usr/bin/env bash

REPO="/dfs/scratch0/kexiao/Github/ukb-cardiac-mri/ukb"
TRAINER=${REPO}/"predict.py"

BATCH_SIZE=8
N_FRAMES=10
IMAGE_TYPE="rgb"
DATA_CONFIG="normalized_var_std"

TEST="/lfs/1/heartmri/test32"
LABELCSV="${TEST}/labels.csv"
MODEL_NAME="Dense4012FrameRNN"
MODEL_WEIGHTS_PATH="Supervised/results_F"

HOME_FOLDER=${1}
SEED=${2}
EXTRA_COMMAND=${3}

DATA_CONFIG=${REPO}/"configs/data/${DATA_CONFIG}.json"

mkdir -p ${HOME_FOLDER}/out
mkdir -p ${HOME_FOLDER}/err
mkdir -p ${HOME_FOLDER}/results

python ${TRAINER} --test ${TEST} \
                  --labelcsv ${LABELCSV} \
                  --model_name ${MODEL_NAME} \
                  --model_weights_path ${MODEL_WEIGHTS_PATH} \
                  -B ${BATCH_SIZE} \
                  -F ${N_FRAMES} \
                  -I ${IMAGE_TYPE} \
                  -a ${DATA_CONFIG} \
                  --seed ${SEED} \
                  --report \
                  --use_cuda \
                  --verbose \
                  --data_seed 2018 \
                  --outdir ${HOME_FOLDER}/results \
                  ${EXTRA_COMMAND} \
                  1>${HOME_FOLDER}/out/seed_${SEED}.out \
                  2>${HOME_FOLDER}/err/seed_${SEED}.err
