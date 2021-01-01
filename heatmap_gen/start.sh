#!/bin/bash

source ../conf/variables.sh

rm -rf ${OUT_DIR}/json \
	${OUT_DIR}/patch-level-lym \
	${OUT_DIR}/patch-level-nec \
	${OUT_DIR}/patch-level-color \
	${OUT_DIR}/patch-level-merged
mkdir  ${OUT_DIR}/json \
	${OUT_DIR}/patch-level-lym \
	${OUT_DIR}/patch-level-nec \
	${OUT_DIR}/patch-level-color \
	${OUT_DIR}/patch-level-merged

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
bash cp_heatmaps_all.sh ${PATCH_PATH} > ${LOG_OUTPUT_FOLDER}/log.cp_heatmaps_all.txt 2>&1

# Combine patch-level and necrosis heatmaps into one heatmap.
# Also generate high-res and low-res version.
bash combine_lym_necrosis_all.sh > ${LOG_OUTPUT_FOLDER}/log.combine_lym_necrosis_all.txt 2>&1
rm ${HEATMAP_TXT_OUTPUT_FOLDER}/*
cp ${OUT_DIR}/patch-level-merged/* ${HEATMAP_TXT_OUTPUT_FOLDER}/     #/data/heatmap_txt
cp ${OUT_DIR}/patch-level-color/* ${HEATMAP_TXT_OUTPUT_FOLDER}/      #/data/heatmap_txt

# Generate meta and heatmap files for high-res and low-res heatmaps.
bash gen_all_json.sh > ${LOG_OUTPUT_FOLDER}/log.gen_all_json.txt 2>&1
cp ${OUT_DIR}/json/* ${JSON_OUTPUT_FOLDER}/

exit 0
