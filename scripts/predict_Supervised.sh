
#################################################################
# Generating Dev results
#################################################################

TEST="data/dev32_gen"
LABELCSV="data/dev32_gen/labels.csv"
MODEL_NAME="Dense4012FrameRNN"
MODEL_WEIGHTS_PATH="Supervised/results"
DEV_FOLDER="Supervised/predictions/results_dev"
SEEDS="0 14 57 123 1234"

COMMAND="--test ${TEST}
         --labelcsv ${LABELCSV}
         --model_name ${MODEL_NAME}
         --model_weights_path ${MODEL_WEIGHTS_PATH}"

for SEED in ${SEEDS}; do
    ./scripts/predict_base.sh "${DEV_FOLDER}" "${SEED}" "${COMMAND}"
done

#################################################################
# Generating Test results
#################################################################

TEST="data/test32_gen"
LABELCSV="data/test32_gen/labels.csv"
MODEL_NAME="Dense4012FrameRNN"
MODEL_WEIGHTS_PATH="Supervised/results"
RESULTS_FOLDER="Supervised/predictions/results_test"
SEEDS="0 14 57 123 1234"

COMMAND="--test ${TEST}
         --labelcsv ${LABELCSV}
         --model_name ${MODEL_NAME}
         --model_weights_path ${MODEL_WEIGHTS_PATH}"

for SEED in ${SEEDS}; do
    ./scripts/predict_base.sh "${RESULTS_FOLDER}" "${SEED}" "${COMMAND}"
done

#################################################################
# Generating Test Ensemble results
#################################################################
python ukb/ensemble.py --results_dir ${RESULTS_FOLDER} \
                       --dev_dir ${DEV_FOLDER} \
                       --metric "f1_score"
