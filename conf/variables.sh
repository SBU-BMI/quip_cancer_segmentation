#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
MONGODB_HOST=xxx
MONGODB_PORT=27017
CANCER_TYPE=brca

if [[ -z "${HEATMAP_VERSION}" ]]; then
	export HEATMAP_VERSION=cancer-brca
fi

# Base directory
export BASE_DIR=/quip_app/quip_cancer_segmentation
export DATA_DIR=/data
export OUT_DIR=${DATA_DIR}/output

# Prediction folders
# Paths of data, log, input, and output
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export SVS_INPUT_PATH=${DATA_DIR}/svs
export PATCH_PATH=${DATA_DIR}/patches

# model is in ${LYM_NECRO_CNN_MODEL_PATH} 
export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
if [[ -z "${CNN_MODEL}" ]]; then
	MODEL="RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7"
else
	if [ "${CNN_MODEL}" == "inceptionv4" ]; then
		MODEL="inceptionv4_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_0423_0449_0.8854108440469536_11.t7"
	elif [ "${CNN_MODEL}" == "vgg16" ]; then
		MODEL="VGG16_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_0423_0456_0.8766822301565503_11.t7"
	elif [ "${CNN_MODEL}" == "resnet34" ]; then
		MODEL="RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7"
	else
		MODEL="RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7"
	fi     
fi
export MODEL

# VERSION INFO
export MODEL_PATH=$LYM_NECRO_CNN_MODEL_PATH/$MODEL 
export TUMOR_VERSION=$(git show --oneline -s | cut -f 1 -d ' ')":"$MODEL_VER":"$(sha256sum $MODEL_PATH | cut -c1-7)
export HEATMAP_VERSION=$HEATMAP_VERSION":"$TUMOR_VERSION
export GIT_REMOTE=$(git remote -v | head -n 1 | cut -f 1 -d ' '| cut -f 2)
export GIT_BRANCH=$(git branch | grep "\*" | cut -f 2 -d ' ')
export GIT_COMMIT=$(git show | head -n 1 | cut -f 2 -d ' ')
export MODEL_HASH=$(sha256sum $MODEL_PATH | cut -f 1 -d ' ')

# Training folders
# The list of case_ids you want to download heaetmaps from
export CASE_LIST=${DATA_DIR}/raw_marking_to_download_case_list/case_list.txt
export DATA_PATH=${DATA_DIR}/training_data   # Change this to your training data folder
export DATA_LIST="tumor_data_list_toy.txt"        # Text file to contain subfolders for testing (1st line), training (the rest)
export PATCH_SAMPLING_LIST_PATH=${DATA_DIR}/patch_sample_list
export RAW_MARKINGS_PATH=${DATA_DIR}/raw_marking_xy
export MODIFIED_HEATMAPS_PATH=${DATA_DIR}/modified_heatmaps
export TUMOR_HEATMAPS_PATH=${DATA_DIR}/tumor_labeled_heatmaps
export TUMOR_GROUND_TRUTH=${DATA_DIR}/tumor_ground_truth_maps
export TUMOR_IMAGES_TO_EXTRACT=${DATA_DIR}/tumor_images_to_extract
export GRAYSCALE_HEATMAPS_PATH=${DATA_DIR}/grayscale_heatmaps
export THRESHOLDED_HEATMAPS_PATH=${DATA_DIR}/thresholded_heatmaps
export PATCH_FROM_HEATMAP_PATH=${DATA_DIR}/patches_from_heatmap
export THRESHOLD_LIST=${DATA_DIR}/threshold_list/threshold_list.txt
export LYM_CNN_TRAINING_DATA=${DATA_DIR}/training_data_cnn

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	export LYM_CNN_TRAINING_DEVICE=0
	export LYM_CNN_PRED_DEVICE=0
else
	export LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

