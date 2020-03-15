#!/bin/bash

NUM_RUNS=$1
CURRENT_FOL=$PWD
source ./0_setup_softlinks.sh
rm ${CODE_DIR}/data/log/log.color.txt
rm ${CODE_DIR}/data/log/log.cnn.txt
rm ${CODE_DIR}/data/log/log.prediction.txt
cd ${CURRENT_FOL}

for (( i=1; i<=${NUM_RUNS}; i++ ))
do
    TMP="tmp.lsf"
    echo "#!/bin/bash -x" > ${TMP}
    echo "#BSUB -P med108" >> ${TMP}
    echo "#BSUB -J brca_prediction_${i}" >> ${TMP}
    echo "#BSUB -o ./logs/log.brca_prediction.o%J" >> ${TMP}
    echo "#BSUB -e ./logs/log.brca_prediction.e%J" >> ${TMP}
    echo "#BSUB -W 02:00" >> ${TMP}
    echo "#BSUB -B" >> ${TMP}
    echo "#BSUB -alloc_flags \"smt4\"" >> ${TMP}
    echo "#BSUB -nnodes 1" >> ${TMP}
    echo "source ./0_activate_environment.sh" >> ${TMP}
    echo "source ./0_user_input.sh" >> ${TMP}
    echo "cd \${CODE_DIR}/prediction" >> ${TMP}
    echo "jsrun -n 1 -a 1 -c 16 -g 2 -b rs ./start.sh" >> ${TMP}

    bsub ${TMP}
    rm -f ${TMP}
done
