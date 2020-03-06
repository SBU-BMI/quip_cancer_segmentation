#!/bin/bash

# *****User input: change location for svs folder and output folder
SVS_DIR="/gpfs/alpine/med108/proj-shared/hdle/data_sample_brca/svs"
OUT_DIR="/gpfs/alpine/med108/proj-shared/hdle/data_sample_brca_out"
CODE_DIR="/gpfs/alpine/med108/proj-shared/hdle/quip_cancer_segmentation"
# end of user input


#setup environment
source /gpfs/alpine/med108/proj-shared/programs/source.sh
export PATH="/ccs/home/hdle/anaconda3/bin:$PATH"
source activate /ccs/home/hdle/anaconda3/envs/py3_torch04

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
