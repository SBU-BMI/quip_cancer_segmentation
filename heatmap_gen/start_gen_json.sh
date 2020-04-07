#!/bin/bash

source ../conf/variables.sh

echo ${HEATMAP_TXT_OUTPUT_FOLDER}
# Generate meta and heatmap files
python -u start_gen_json.py ${HEATMAP_TXT_OUTPUT_FOLDER} ${SVS_INPUT_PATH} ${HEATMAP_VERSION} ${LOG_OUTPUT_FOLDER} > ${LOG_OUTPUT_FOLDER}/log.start_gen_json.txt

wait;
mv json/* ${JSON_OUTPUT_FOLDER}

