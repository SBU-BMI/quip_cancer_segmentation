# quip_cancer_segmentation

This repo is for training and testing brca cancer detection pipeline using 3 standard CNNs: VGG16, Resnet-34, and Inception-v4. 
More details are found in the paper: [Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer](https://arxiv.org/abs/1905.10841)

NOTE: download the trained models [here](https://drive.google.com/open?id=1km7gVpBpLbBovExTgt3CE8JRwpTEl57F), extract 3 *.t7  files to data/models_cnn

The default settings are for Resnet-34 since it performs the best on the public testset. To use other models, change the variable "MODEL" in conf/variables.sh to other models name downloaded from google drive above.

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
 
 More details are in file brca_environ.txt

# Running Codes Instrucstions
- Codes are in folder scripts, including training and testing
- Need to setup folder path for training data, model. All parameters are found in conf/variables.sh

## Setup conf/variables.sh
- Change the BASE_DIR to the path of your folder after you clone the git repo

## Training:
- Settings are stored in conf/variables.sh
- Change DATA_PATH to your folder that contains all subfolders for training
- Change DATA_LIST to your text file name that contains list of subfolders for training and validataion. 1st line is for validation, the rest is for training. Example of the list is tumor_data_list_toy.txt 
- Run a demo training that uses a subset of training data: python train_cancer_cnn_Resnet_pretrained.py  
- To run a full training that uses all training data, remove line 107-111 in "train_cancer_cnn_Resnet_pretrained.py", then run python train_cancer_cnn_Resnet_pretrained.py
- Log files are in data/log
- Trained models are in "checkpoint"

## Testing
#### Preparation:
- Change MODEL in conf/variables.sh to your model name that is stored in data/models_cnn
- Copy all .svs files to data/svs
  + For example, cd to your data/svs, run "cp /data01/shared/hanle/svs_tcga_seer_brca/TCGA-3C-AALI-01Z-00-DX2.svs ."
#### Process .svs files (processing in computational nodes):
- Patch extraction: go to folder "patch_extraction_cancer_40X", run "nohup bash start.sh &"
- Prediction: go to folder "prediction", run "nohup bash start.sh &"
- Generate json heatmap files: go to folder "heatmap_gen", run "nohup bash start.sh &"
  + Output are in data/heatmap_txt and data/heatmap_jsons
    
#### 1 script to run all: 
- To run all the above steps, including patch extraction, prediction, and generate json files, go to folder "scripts", run bash svs_2_heatmap.sh

#### Generate Grayscale heatmap: 
  + Go to folder "download_heatmap/get_grayscale_heatmaps", run "bash start.sh"
  + Results are stored at download_heatmap/get_grayscale_heatmaps/grayscale_heatmaps and data/grayscale_heatmaps

#### Confirm the results:
  + Compare the grayscale heatmap with the one on website: [https://mathbiol.github.io/tcgatil/](https://mathbiol.github.io/tcgatil/)


# Docker Instructions 

A Docker image is available at: [pytorch docker](https://cloud.docker.com/repository/docker/hanle/brca-pipeline-image)
## Step 1:
Create folder named "data" and subfoders below:

- change the BASE_DIR setting in conf/variables.sh to the path of your working directory
- data/svs: to contains *.svs files
- data/training_data: to contain training data
- data/patches: to contain output from patch extraction
- data/log: to contain log files
- data/heatmap_txt: to contain prediction output

## Step 2:
- Run "bash create_container.sh" to create container for the docker
- Run "bash start_interactive_bash.sh" to start the docker workspace
- Clone codes from this repository to workspace of docker.
- run: "mv quip_cancer_segmentation/* ."
- Follow instructions for Training and Testing as below.


# Citation
     @article{le2020utilizing,
       title={Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer},
       author={Le, Han and Gupta, Rajarsi and Hou, Le and Abousamra, Shahira and Fassler, Danielle and Torre-Healy, Luke and Moffitt, Richard A and Kurc, Tahsin and Samaras, Dimitris and Batiste, Rebecca and others},
       journal={The American Journal of Pathology},
       year={2020},
       publisher={Elsevier}
      }
