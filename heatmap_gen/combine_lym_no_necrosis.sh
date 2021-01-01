#!/bin/bash

source ../conf/variables.sh

FN=$1
LYM_FOLDER=${OUT_DIR}/patch-level-lym/
OUT_FOLDER=${OUT_DIR}/patch-level-merged/

awk '{
    print $1, $2, $3, 0.0;
}' ${LYM_FOLDER}/${FN} > ${OUT_FOLDER}/${FN} 2>&1

exit 0

