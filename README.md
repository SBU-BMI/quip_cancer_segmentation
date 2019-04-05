# quip_cancer_segmentation

This repo is for training and testing brca cancer detection pipeline.

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Numpy
 - [OpenCV-Python](https://pypi.python.org/pypi/opencv-python)
 - Openslide 1.1.1 (https://openslide.org/api/python/)
 - sklearn (https://scikit-learn.org/stable/)
 - PIL (https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
 
 Details are in file brca_environ.txt

# Running codes
- Codes are in folder scripts, including training and testing
- Need to setup folder path for training data, model. All parameters are found in conf/variables.sh
- Training: bash train.sh
-- 
- Testing: bash svs_2_heatmap.sh
