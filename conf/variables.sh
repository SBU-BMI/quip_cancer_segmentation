#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=quip1.bmi.stonybrook.edu
MONGODB_PORT=27017
HEATMAP_VERSION=cancer_BC_v2

# Base directory
BASE_DIR=/data10/shared/hanle/quip_cancer_segmentation  # change this to your current working path

# The list of case_ids you want to download heaetmaps from
CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt
DATA_PATH=${BASE_DIR}/data/training_data        # Change this to your training data folder
DATA_LIST='tumor_data_list_toy.txt'        # Text file to contain subfolders for testing (1st line), training (the rest)
MODEL='RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_cancer_tils_none_1117_1811_0.9157633018398808_9.t7'     # model is saved in data/models_cnn


# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_txt
LOG_OUTPUT_FOLDER=${BASE_DIR}/data/log
SVS_INPUT_PATH=${BASE_DIR}/data/svs
PATCH_PATH=${BASE_DIR}/data/patches
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
LYM_CNN_TRAINING_DEVICE=0
LYM_CNN_PRED_DEVICE=0
