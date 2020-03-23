#!/bin/bash

source ../conf/variables.sh

cd ./utils
echo "Usage: pass 1 to the argument to display more detail info. e.g: bash $0 1"
echo "Counting files..."
echo ""
python -u check_if_codes_done.py ${SVS_INPUT_PATH} ${PATCH_PATH} ${HEATMAP_TXT_OUTPUT_FOLDER} ${JSON_OUTPUT_FOLDER} $1
