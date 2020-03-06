#!/bin/bash
#-d pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

mkdir data
mkdir data/svs data/patches data/log data/heatmap_txt data/training_data

nvidia-docker run \
    --shm-size=8G \
    --name han-pytorch -it \
    -v `pwd`/data/:/home/data/ \
    -d hanle/brca-pipeline-image
exit 0
