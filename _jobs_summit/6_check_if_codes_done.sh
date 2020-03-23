#!/bin/bash

source ../conf/variables.sh

cd ./utils
python -u check_if_codes_done.py ${SVS_INPUT_PATH} ${PATCH_PATH} ${HEATMAP_TXT_OUTPUT_FOLDER} ${JSON_OUTPUT_FOLDER}
