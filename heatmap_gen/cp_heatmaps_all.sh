#!/bin/bash

FOLDER=${1}     #/data/patches

PRED_VERSION=patch-level-cancer.txt
DIS_FOLDER=./patch-level-lym/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1)));}'`

    SVS_EXTENSION=`echo ${files} | awk -F'/' '{print $(NF-1)}' | awk -F'.' '{print $NF}'`
    len_dis=`expr ${#dis} - ${#SVS_EXTENSION} - 1`
    dis=${dis:0:${len_dis}}
    cp ${files} ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-color-cancer.txt
DIS_FOLDER=./patch-level-color/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "color-"substr($(NF-1),1,length($(NF-1)));}'`

    SVS_EXTENSION=`echo ${files} | awk -F'/' '{print $(NF-1)}' | awk -F'.' '{print $NF}'`
    len_dis=`expr ${#dis} - ${#SVS_EXTENSION} - 1`
    dis=${dis:0:${len_dis}}
    cp ${files} ${DIS_FOLDER}/${dis}
done

rm '~/running_cp_heatmap.txt'

exit 0
