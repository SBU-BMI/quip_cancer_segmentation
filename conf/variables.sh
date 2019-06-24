#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=xyz.bmi.stonybrook.edu
MONGODB_PORT=27017

if [[ -z "${HEATMAP_VERSION}" ]]; then
	HEATMAP_VERSION=cancer_BC_v2
fi

# Base directory
BASE_DIR=/root/quip_cancer_segmentation  # change this to your current working path
HOST_DIR=/data

# The list of case_ids you want to download heatmaps from
CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt
DATA_PATH=${BASE_DIR}/data/training_data  # Change this to your training data folder
DATA_LIST='tumor_data_list_toy.txt'       # Text file to contain subfolders for testing (1st line), training (the rest)

# model is in data/models_cnn
if [[ -z "${MODEL}" ]]; then
	MODEL='RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7'     
else
	if [ "${MODEL}" == "inceptionv4" ]; then
		MODEL='inceptionv4_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_0423_0449_0.8854108440469536_11.t7'
	elif [ "${MODEL}" == "vgg16" ]; then
		MODEL='VGG16_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_0423_0456_0.8766822301565503_11.t7'
	elif [ "${MODEL}" == "resnet34" ]; then
		MODEL='RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7'
	else
		MODEL='RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7'
	fi     
fi

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${HOST_DIR}/output/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${HOST_DIR}/output/heatmap_txt
LOG_OUTPUT_FOLDER=${HOST_DIR}/output/log
SVS_INPUT_PATH=${HOST_DIR}/svs
PATCH_PATH=${HOST_DIR}/patches

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

LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/data/models_cnn
LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn

if [[ -z "${CUDA_VISIBLE_DEVICE}" ]]; then
	LYM_CNN_TRAINING_DEVICE=0
	LYM_CNN_PRED_DEVICE=0
else
	LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICE}
	LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICE}
fi
