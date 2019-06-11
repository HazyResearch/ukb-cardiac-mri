#!/usr/bin/env bash

FOLDER="WeakSupervised"

DATA_CONFIG="normalized_var_std"
MODEL_CONFIG="dn40_lstm_min"
SAMPLE_TYPE=1

#N_SAMPLES="100 500 1000 2000 4000 8000"
N_SAMPLES="100"

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
                          "${EXTRA_COMMAND}" &
    done
done

