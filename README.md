# quip_cancer_segmentation

This repo is for training and testing brca cancer detection pipeline.

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Numpy
 - [OpenCV-Python](https://pypi.python.org/pypi/opencv-python)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
 
 Details are in file brca_environ.txt

# Running codes
- Codes are in folder scripts, including training and testing
- Need to setup folder path for training data, model. All parameters are found in conf/variables.sh
## Training:
- Change DATA_PATH to your folder that contains all subfolders for training
- Change DATA_LIST to your text file name that contains list of subfolders for training and validataion. 1st line is for validation, the rest is for training. Example of the list is tumor_training_list_toy.txt 
- Run "bash train.sh"
- Log files are in data/log
- Trained models are in "checkpoint"

## Testing
- Change MODEL to your model name that is stored in data/models_cnn
- Copy all .svs files to data/svs
- Run "bash svs_2_heatmap.sh"
- Output are in data/heatmap_txt
