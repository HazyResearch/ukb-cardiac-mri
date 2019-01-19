#!/usr/bin/env bash

REPO="ukb"
TRAINER=${REPO}/"train.py"
TRAIN="/lfs/1/heartmri/coral32"
DEV="/lfs/1/heartmri/dev32"
SEMI="/lfs/1/heartmri/semi32"
SEMI_CSV="labels_train.csv"

BATCH_SIZE=8
N_EPOCHS=20
N_MODELS=12
UPDATE_FREQ=1
CKPNT_FREQ=10
IMAGE_TYPE="rgb"
SERIES=0
FRAMES=10
METRIC="f1_score"

HOME_FOLDER=${1}
MODEL_CONFIG=${2}
DATA_CONFIG=${3}
SEED=${4}
SAMPLE_TYPE=${5}
N_SAMPLES=${6}
EXTRA_COMMAND=${7}

DATA_CONFIG="${REPO}/configs/data/${DATA_CONFIG}.json"
MODEL_CONFIG="${REPO}/configs/ukbb/${MODEL_CONFIG}.json"

if [ -z "${8}" ]; then
    SEMI_CMD=""
else
    SEMI_CMD="--semi --semi_dir ${SEMI} --semi_csv ${SEMI_CSV} --sample_split 0.074"
fi

mkdir -p ${HOME_FOLDER}/out
mkdir -p ${HOME_FOLDER}/err
mkdir -p ${HOME_FOLDER}/results


printf "\n${TRAINER}
                        --train ${TRAIN}
                        --dev ${DEV}
                        --test ${DEV}
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
                        --seed ${SEED}
                        -R
                        --report
                        --use_cuda
                        --verbose
                        --pretrained
                        --requires_grad
                        --noise_aware
                        --sample
                        --data_seed 2018
                        --data_threshold 0.75
                        --sample_type ${SAMPLE_TYPE}
                        --n_samples ${N_SAMPLES}
                        ${EXTRA_COMMAND}
                        ${SEMI_CMD}
                        1>${HOME_FOLDER}/out/seed_${SEED}.out
                        2>${HOME_FOLDER}/err/seed_${SEED}.err\n"

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
                  --seed ${SEED} \
                  -R \
                  --report \
                  --use_cuda \
                  --verbose \
                  --pretrained \
                  --requires_grad \
                  --noise_aware \
                  --sample \
                  --data_seed 2018 \
                  --data_threshold 0.75 \
                  --sample_type ${SAMPLE_TYPE} \
                  --n_samples ${N_SAMPLES} \
                  ${EXTRA_COMMAND} \
                  ${SEMI_CMD} \
                  1>${HOME_FOLDER}/out/seed_${SEED}.out \
                  2>${HOME_FOLDER}/err/seed_${SEED}.err 
