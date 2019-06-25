# quip_cancer_segmentation

This repo is for running the BRCA (breast cancer) cancer detection pipeline using one of the 3 standard CNNs: 
VGG16, Resnet-34, and Inception-v4. More details are found in the paper: [Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer](https://arxiv.org/abs/1905.10841)

# Docker Instructions 

Build the docker image by: 

docker build -t cancer_prediction .  (Note the dot at the end). 

## Step 1:
Create folder named "data" and subfoders below on the host machine:

- data/svs: to contains *.svs files
- data/patches: to contain output from patch extraction
- data/log: to contain log files
- data/heatmap_txt: to contain prediction output
- data/heatmap_jsons: to contain prediction output as json files

## Step 2:
- Run the docker container as follows: 

nvidia-docker run --name cancer_prediction_pipeline -itd -v <path-to-data>:/data -e MODEL="<model>" -e HEATMAP_VERSION="<heatmap version>" -e CUDA_VISIBLE_DEVICES='<cuda device id>' -e CANCER_TYPE="breast" cancer_prediction svs_2_heatmap.sh 
 
MODEL -- the CNN model to use. It can be resnet34, inceptionv4, or vgg16
HEATMAP_VERSION -- used to set the analysis execution id of the run (for uploading to the database)
CUDA_VISIBLE_DEVICES -- set to select the GPU to use 
CANCER_TYPE -- set the cancer type. 

The following example runs the cancer detection pipeline using the ResNet-34 model on GPU 0. It will process images in /home/user/data/svs and output the results to /home/user/data/output. 

nvidia-docker run --name cancer_prediction_pipeline -itd -v /home/user/data:/data -e MODEL="resnet34" -e HEATMAP_VERSION="resnet_v1" -e CUDA_VISIBLE_DEVICES='0' -e CANCER_TYPE="breast" cancer_prediction svs_2_heatmap.sh

# Citation
     @article{le2019utilizing,
       title={Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer},
       author={Le, Han and Gupta, Rajarsi and Hou, Le and Abousamra, Shahira and Fassler, Danielle and Kurc, Tahsin and Samaras, Dimitris and Batiste, Rebecca and Zhao, Tianhao and Van Dyke, Alison L and Sharma, Ashish and Bremer, Erich and Almeida, Jonas S. and Saltz, Joel},
       journal={arXiv preprint arXiv:1905.10841},
       year={2019}
     }
