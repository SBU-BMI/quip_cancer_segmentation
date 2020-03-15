#!/bin/bash

source ./0_user_input.sh

# setup soflinks to svs folder and output folder, no change required from users
mkdir ${OUT_DIR}
cd ${OUT_DIR}
mkdir data
cd data
mkdir patches log heatmap_txt heatmap_jsons
rm svs  # remove softlink to svs if exists
ln -s ${SVS_DIR} svs    # create new softlink

cd ${CODE_DIR}
rm data # remove softlink to data if exists
ln -s ${OUT_DIR}/data data      # create new softlink
# end of setting softlinks
