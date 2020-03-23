#!/bin/bash

NUM_RUNS=$1
CURRENT_FOL=$PWD
source 0_user_input.sh
rm ${CODE_DIR}/data/log/log.save_svs_to_tiles.txt
cd ${CURRENT_FOL}

for (( i=1; i<=${NUM_RUNS}; i++ ))
do
    TMP="tmp.lsf"
    echo "#!/bin/bash -x" > ${TMP}
    echo "#BSUB -P med108" >> ${TMP}
    echo "#BSUB -J brca_patch_extraction_${i}" >> ${TMP}
    echo "#BSUB -o ./logs/log.brca_patch_extraction.o%J" >> ${TMP}
    echo "#BSUB -e ./logs/log.brca_patch_extraction.e%J" >> ${TMP}
    echo "#BSUB -W 02:00" >> ${TMP}
    echo "#BSUB -B" >> ${TMP}
    echo "#BSUB -alloc_flags \"smt4\"" >> ${TMP}
    echo "#BSUB -nnodes 1" >> ${TMP}
    echo "source ./utils/0_activate_environment.sh" >> ${TMP}
    echo "source 0_user_input.sh" >> ${TMP}
    echo "cd \${CODE_DIR}/patch_extraction_cancer_40X" >> ${TMP}
    echo "jsrun -n 1 -a 1 -c 42 -g 0 -b rs ./start.sh" >> ${TMP}

    bsub ${TMP}
    rm -f ${TMP}
done

