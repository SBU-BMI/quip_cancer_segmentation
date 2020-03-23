# Running quip_cancer_segmentation on SUMMIT

IMPORTANT: Download the trained models [here](https://drive.google.com/open?id=1km7gVpBpLbBovExTgt3CE8JRwpTEl57F), extract 3 *.t7  files to ./models_cnn

The default settings are for Resnet-34 since it performs the best on the public testset. To use other models, change the variable "MODEL" in conf/variables.sh to other models name downloaded from google drive above.

## Instructions:

- Jobs are found at ./_jobs_summit
- Specify svs folder and output folder in _jobs_summit/0_user_input.sh: SVS_DIR and OUT_DIR
    - SVS_DIR contains the WSIs that you want to process
    - OUT_DIR contains the results (a folder named "data" will be automatically created inside OUT_DIR).
    - If OUT_DIR does not exists, the script will create a new foder named OUT_DIR
- Run "bash 1_start_setup_softlinks_folders.sh" to setup softlinks, setup folders. Run this command first, only 1 time.
- Run "bash 2_start_patch_extraction.sh ${number_of_jobs}" to start patch extraction. We can submit multiple jobs that will be processed in parallel. ${number_of_jobs} is less than 100 (Summit limit per user).
- Run "bash 3_start_prediction.sh ${number_of_jobs}" to start the prediction. ${number_of_jobs} is set similar as above.
- Run "bash 4_start_cp_heatmaps.sh" to copy heatmap_txt from different folders to a same folder.
- Run "bash 5_start_gen_json.sh ${number_of_jobs}" to start generating json files. ${number_of_jobs} is set similar as above.
- Run "bash 6_check_if_codes_done.sh" at anytime to check if the WSIs are done processing at any steps
- Users can configure the job setting in the "jsrun" command inside each .sh file.

### Example of using scripts
- "bash 1_start_setup_softlinks_folders.sh"
- Run patch extraction and prediction in parallel, use 50 jobs for each. This number depends on availble resources on Summit. To see the availble resources, run "jobstat -u ${your_username}":
    - "bash 2_start_patch_extraction.sh 50"
    - "bash 3_start_prediction.sh 50"
- Wait for these 2 steps done running. Then proceed the next steps below sequentially. 
- "bash 4_start_cp_heatmaps.sh"
- "bash 5_start_gen_json.sh 20"


# Citation
     @article{le2019utilizing,
       title={Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer},
       author={Le, Han and Gupta, Rajarsi and Hou, Le and Abousamra, Shahira and Fassler, Danielle and Kurc, Tahsin and Samaras, Dimitris and Batiste, Rebecca and Zhao, Tianhao and Van Dyke, Alison L and Sharma, Ashish and Bremer, Erich and Almeida, Jonas S. and Saltz, Joel},
       journal={arXiv preprint arXiv:1905.10841},
       year={2019}
     }
