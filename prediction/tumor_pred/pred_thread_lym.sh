#!/bin/bash

source ../../conf/variables.sh

FOLDER=$1
# PARAL = [0, MAX_PARAL-1]
PARAL=$2
MAX_PARAL=$3
DEVICE=$4

DATA_FILE=patch-level-cancer.txt
DONE_FILE=extraction_done.txt
EXEC_FILE=pred.py

PRE_FILE_NUM=0
while [ 1 ]; do
    LINE_N=0
    FILE_NUM=0
    EXTRACTING=0
    for files in ${FOLDER}/*/; do
        FILE_NUM=$((FILE_NUM+1))
        if [ ! -f ${files}/${DONE_FILE} ]; then EXTRACTING=1; fi

        LINE_N=$((LINE_N+1))
        if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

        if [ -f ${files}/${DONE_FILE} ]; then
            if [ ! -f ${files}/${DATA_FILE} ]; then
                echo ${files}/${DATA_FILE} generating
                python -u ${EXEC_FILE} ${files} ${LYM_NECRO_CNN_MODEL_PATH} ${DATA_FILE} ${LYM_NECRO_CNN_MODEL_PATH}/${MODEL}
            fi
        fi
    done

    if [ ${EXTRACTING} -eq 0 ] && [ ${PRE_FILE_NUM} -eq ${FILE_NUM} ]; then break; fi
    PRE_FILE_NUM=${FILE_NUM}
done

exit 0
