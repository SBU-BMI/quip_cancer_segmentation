#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
MONGODB_HOST=xyz
MONGODB_PORT=27017

if [[ -z "${CANCER_TYPE}" ]]; then
	CANCER_TYPE=undefined
fi

if [[ -z "${HEATMAP_VERSION}" ]]; then
	HEATMAP_VERSION=BC_til_resnet
fi

# Base directory
BASE_DIR=/root/ajp_til_analysis
HOST_DIR=/data

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${HOST_DIR}/output/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${HOST_DIR}/output/heatmap_txt
LOG_OUTPUT_FOLDER=${HOST_DIR}/output/log
SVS_INPUT_PATH=${HOST_DIR}/svs
PATCH_PATH=${HOST_DIR}/patches

# TIL prediction model
LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
MODEL='RESNET_34_LYM_200px_lr_1e-2_decay_5_jitter_none_0416_1549_0.8716094032549727_15.t7' 

# The list of case_ids you want to download heaetmaps from
CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt
DATA_PATH=${BASE_DIR}/data/training_data        # Change this to your training data folder
DATA_LIST='tumor_data_list_toy.txt'        # Text file to contain subfolders for testing (1st line), training (the rest)
PATCH_SAMPLING_LIST_PATH=${BASE_DIR}/data/patch_sample_list
RAW_MARKINGS_PATH=${BASE_DIR}/data/raw_marking_xy
MODIFIED_HEATMAPS_PATH=${BASE_DIR}/data/modified_heatmaps
TUMOR_HEATMAPS_PATH=${BASE_DIR}/data/tumor_labeled_heatmaps
TUMOR_GROUND_TRUTH=${BASE_DIR}/data/tumor_ground_truth_maps
TUMOR_IMAGES_TO_EXTRACT=${BASE_DIR}/data/tumor_images_to_extract
GRAYSCALE_HEATMAPS_PATH=${BASE_DIR}/data/grayscale_heatmaps
THRESHOLDED_HEATMAPS_PATH=${BASE_DIR}/data/thresholded_heatmaps
PATCH_FROM_HEATMAP_PATH=${BASE_DIR}/data/patches_from_heatmap
THRESHOLD_LIST=${BASE_DIR}/data/threshold_list/threshold_list.txt
LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn

EXTERNAL_LYM_MODEL=0

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	LYM_CNN_TRAINING_DEVICE=0
	LYM_CNN_PRED_DEVICE=0
else
	LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

