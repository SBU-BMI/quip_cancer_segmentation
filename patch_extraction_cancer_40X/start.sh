#!/bin/bash

source ../conf/variables.sh

nohup python -u start.py ${SVS_INPUT_PATH} ${PATCH_PATH} ${LOG_OUTPUT_FOLDER} &
