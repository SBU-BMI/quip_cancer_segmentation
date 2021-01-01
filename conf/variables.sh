#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
MONGODB_HOST=xxx
MONGODB_PORT=27017
CANCER_TYPE=brca

if [[ -z "${HEATMAP_VERSION}" ]]; then
	export HEATMAP_VERSION=cancer_BC_v2
fi

# Base directory
export BASE_DIR=/root/quip_cancer_segmentation
export DATA_DIR=${BASE_DIR}/data
export OUT_DIR=${BASE_DIR}/output

# Paths of data, log, input, and output
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export SVS_INPUT_PATH=${DATA_DIR}/svs
export PATCH_PATH=${DATA_DIR}/patches

# The list of case_ids you want to download heaetmaps from
export CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt
export DATA_PATH=${BASE_DIR}/data/training_data   # Change this to your training data folder
export DATA_LIST='tumor_data_list_toy.txt'        # Text file to contain subfolders for testing (1st line), training (the rest)

# model is in ${LYM_NECRO_CNN_MODEL_PATH} 
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
export MODEL
export CNN_MODEL=${MODEL}

export PATCH_SAMPLING_LIST_PATH=${BASE_DIR}/data/patch_sample_list
export RAW_MARKINGS_PATH=${BASE_DIR}/data/raw_marking_xy
export MODIFIED_HEATMAPS_PATH=${BASE_DIR}/data/modified_heatmaps
export TUMOR_HEATMAPS_PATH=${BASE_DIR}/data/tumor_labeled_heatmaps
export TUMOR_GROUND_TRUTH=${BASE_DIR}/data/tumor_ground_truth_maps
export TUMOR_IMAGES_TO_EXTRACT=${BASE_DIR}/data/tumor_images_to_extract
export GRAYSCALE_HEATMAPS_PATH=${BASE_DIR}/data/grayscale_heatmaps
export THRESHOLDED_HEATMAPS_PATH=${BASE_DIR}/data/thresholded_heatmaps
export PATCH_FROM_HEATMAP_PATH=${BASE_DIR}/data/patches_from_heatmap
export THRESHOLD_LIST=${BASE_DIR}/data/threshold_list/threshold_list.txt

export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
export LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
EXTERNAL_LYM_MODEL=0

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	export LYM_CNN_TRAINING_DEVICE=0
	export LYM_CNN_PRED_DEVICE=0
else
	export LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

