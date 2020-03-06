# Running quip_cancer_segmentation on SUMMIT

IMPORTANT: Download the trained models [here](https://drive.google.com/open?id=1km7gVpBpLbBovExTgt3CE8JRwpTEl57F), extract 3 *.t7  files to ./models_cnn

The default settings are for Resnet-34 since it performs the best on the public testset. To use other models, change the variable "MODEL" in conf/variables.sh to other models name downloaded from google drive above.

## Instructions:

- Jobs are found at ./_jobs_summit
- Specify svs folder and output folder in _jobs_summit/0_user_input.sh: SVS_DIR and OUT_DIR
    - SVS_DIR contains the WSIs that you want to process
    - OUT_DIR contains the results (a folder named "data" will be automatically created inside OUT_DIR).
    - If OUT_DIR does not exists, the script will create a new foder named OUT_DIR
- Run "bsub 1_patch_extraction.lsf" to start patch extraction
- Run "bsub 2_prediction.lsf" to start the prediction
- Run "bsub 3_heatmap_gen.lsf" to start generating heatmap jsons
- Users can configure the job setting in the "jsrun" command.


# Citation
     @article{le2019utilizing,
       title={Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer},
       author={Le, Han and Gupta, Rajarsi and Hou, Le and Abousamra, Shahira and Fassler, Danielle and Kurc, Tahsin and Samaras, Dimitris and Batiste, Rebecca and Zhao, Tianhao and Van Dyke, Alison L and Sharma, Ashish and Bremer, Erich and Almeida, Jonas S. and Saltz, Joel},
       journal={arXiv preprint arXiv:1905.10841},
       year={2019}
     }
