#!/bin/bash

source ../conf/variables.sh

rm -rf json patch-level-lym patch-level-nec patch-level-color patch-level-merged
mkdir  json patch-level-lym patch-level-nec patch-level-color patch-level-merged

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
bash cp_heatmaps_all.sh ${PATCH_PATH} &> ${LOG_OUTPUT_FOLDER}/log.cp_heatmaps_all.txt

bash combine_lym_necrosis_all.sh &> ${LOG_OUTPUT_FOLDER}/log.combine_lym_necrosis_all.txt
rm ${HEATMAP_TXT_OUTPUT_FOLDER}/*
cp ./patch-level-merged/* ${HEATMAP_TXT_OUTPUT_FOLDER}/     #/data/heatmap_txt
cp ./patch-level-color/* ${HEATMAP_TXT_OUTPUT_FOLDER}/      #/data/heatmap_txt

exit 0
