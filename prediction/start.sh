#!/bin/bash

source ../conf/variables.sh

cd tumor_pred
nohup bash pred_thread_lym.sh \
    ${PATCH_PATH} 0 1 ${LYM_CNN_PRED_DEVICE} \
    > ${LOG_OUTPUT_FOLDER}/log.pred_thread_tumor_0.txt 2>&1 &
    
cd ..

cd color
nohup bash color_stats.sh ${PATCH_PATH} 0 2 \
    > ${LOG_OUTPUT_FOLDER}/log.color_stats_0.txt 2>&1 &
nohup bash color_stats.sh ${PATCH_PATH} 1 2 \
    > ${LOG_OUTPUT_FOLDER}/log.color_stats_1.txt 2>&1 &
cd ..

wait

exit 0
