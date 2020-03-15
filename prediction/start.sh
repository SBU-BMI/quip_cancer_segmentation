#!/bin/bash

source ../conf/variables.sh

nohup python -u start.py ${PATCH_PATH} ${LYM_NECRO_CNN_MODEL_PATH}/${MODEL} ${LOG_OUTPUT_FOLDER} > ${LOG_OUTPUT_FOLDER}/log.prediction.txt &
