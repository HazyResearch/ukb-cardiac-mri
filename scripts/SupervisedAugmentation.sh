#!/usr/bin/env bash

REPO="ukb"
TRAINER=${REPO}/"train.py"
TRAIN="data/train32_gen"
DEV="data/dev32_gen"
TEST="data/test32_gen"

BATCH_SIZE=8
N_EPOCHS=100
N_MODELS=12
UPDATE_FREQ=1
CKPNT_FREQ=10
IMAGE_TYPE="rgb"
SERIES=0
FRAMES=10
METRIC="f1_score"

HOME_FOLDER="SupervisedAugmentation"
MODEL_CONFIG="dn40_lstm_min"
DATA_CONFIG="affine_01"
EXTRA_COMMAND=""

if [ -z "${1}" ]; then
    SEED="0 14 57 123 1234"
else
    SEED=${1}
fi

DATA_CONFIG="${REPO}/configs/data/${DATA_CONFIG}.json"
MODEL_CONFIG="${REPO}/configs/ukbb/${MODEL_CONFIG}.json"

mkdir -p ${HOME_FOLDER}/out
mkdir -p ${HOME_FOLDER}/err
mkdir -p ${HOME_FOLDER}/results

for seed in ${SEED}; do
    printf "\n${TRAINER}
                            --train ${TRAIN}
                            --dev ${DEV}
                            --test ${TEST}
                            -c ${MODEL_CONFIG}
                            -a ${DATA_CONFIG}
                            -T ${METRIC}
                            -S ${METRIC}
                            -F ${FRAMES}
                            -B ${BATCH_SIZE}
                            -E ${N_EPOCHS}
                            -I ${IMAGE_TYPE}
                            -N ${N_MODELS}
                            --update_freq ${UPDATE_FREQ}
                            --checkpoint_freq ${CKPNT_FREQ}
                            --series ${SERIES}
                            --outdir ${HOME_FOLDER}/results
                            --seed ${seed}
                            -R
                            --report
                            --use_cuda
                            --verbose
                            --pretrained
                            --requires_grad
                            --data_seed 2018
                            --data_threshold 0.75
                            ${EXTRA_COMMAND}
                            1>${HOME_FOLDER}/out/seed_${seed}.out
                            2>${HOME_FOLDER}/err/seed_${seed}.err\n"
    
    python ${TRAINER} --train ${TRAIN} \
                      --dev ${DEV} \
                      --test ${DEV} \
                      -c ${MODEL_CONFIG} \
                      -a ${DATA_CONFIG} \
                      -T ${METRIC} \
                      -S ${METRIC} \
                      -F ${FRAMES} \
                      -B ${BATCH_SIZE} \
                      -E ${N_EPOCHS} \
                      -I ${IMAGE_TYPE} \
                      -N ${N_MODELS} \
                      --update_freq ${UPDATE_FREQ} \
                      --checkpoint_freq ${CKPNT_FREQ} \
                      --series ${SERIES} \
                      --outdir ${HOME_FOLDER}/results \
                      --seed ${seed} \
                      -R \
                      --report \
                      --use_cuda \
                      --verbose \
                      --pretrained \
                      --requires_grad \
                      --data_seed 2018 \
                      --data_threshold 0.75 \
                      ${EXTRA_COMMAND} \
                      1>${HOME_FOLDER}/out/seed_${seed}.out \
                      2>${HOME_FOLDER}/err/seed_${seed}.err &
done
