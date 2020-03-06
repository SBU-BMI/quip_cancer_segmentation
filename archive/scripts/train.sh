#!/bin/bash

cd ../
source ./conf/variables.sh

nohup python -u train_cancer_cnn_Resnet_pretrained.py --data ${DATA_PATH} --data_list ${DATA_LIST} &> ${LOG_OUTPUT_FOLDER}/log.train.txt & 
cd ../..

exit 0
