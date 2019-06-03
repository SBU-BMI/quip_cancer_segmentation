# quip_cancer_segmentation

This repo is for training and testing brca cancer detection pipeline.

A Docker image is available at: [pytorch docker](https://cloud.docker.com/repository/docker/hanle/brca-pipeline-image)

### Step 1:
Create folder named "data" and subfoders below:

- data/svs: to contains *.svs files
- data/training_data: to contain training data
- data/patches: to contain output from patch extraction
- data/log: to contain log files
- data/heatmap_txt: to contain prediction output

### Step 2:
- Run "bash create_container.sh" to create container for the docker
- Run "bash start_interactive_bash.sh" to start the docker workspace
- Clone codes from this repository to workspace of docker.
- run: "mv quip_cancer_segmentation/* ."
- Follow instructions for Training and Testing as below.

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
 
 More details are in file brca_environ.txt

# Running codes
- Codes are in folder scripts, including training and testing
- Need to setup folder path for training data, model. All parameters are found in conf/variables.sh
## Training:
- Settings are stored in conf/variables.sh
- Change DATA_PATH to your folder that contains all subfolders for training
- Change DATA_LIST to your text file name that contains list of subfolders for training and validataion. 1st line is for validation, the rest is for training. Example of the list is tumor_data_list_toy.txt 
- Run "bash train.sh"
- Log files are in data/log
- Trained models are in "checkpoint"

## Testing
- Change MODEL to your model name that is stored in data/models_cnn
- Copy all .svs files to data/svs
- Run "bash svs_2_heatmap.sh"
- Output are in data/heatmap_txt
