#!/usr/bin/env bash

FOLDER="ExpertWeakSupervised"

DATA_CONFIG="normalized_var_std"
MODEL_CONFIG="dn40_lstm_min"
SAMPLE_TYPE=2

#N_SAMPLES="108 216 432 864 1296 1728 3456 4264"
N_SAMPLES="108"

if [ -z "${1}" ]; then
    SEED="0 14 57 123 1234"
else
    SEED=${1}
fi

EXTRA_COMMAND="--labelcsv ANON_unknown_hack3_gender.csv"

for n_samples in ${N_SAMPLES}; do
    for seed in ${SEED}; do

        ./scaleup_base.sh "${FOLDER}/sample${n_samples}" \
                          "${MODEL_CONFIG}" \
                          "${DATA_CONFIG}" \
                          "${seed}" \
                          "${SAMPLE_TYPE}" \
                          "${n_samples}" \
                          "${EXTRA_COMMAND}" \
                          "SEMI_ON" &
    done
done


