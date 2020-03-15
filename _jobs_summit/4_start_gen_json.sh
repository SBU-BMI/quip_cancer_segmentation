#!/bin/bash

NUM_RUNS=$1

for (( i=1; i<=${NUM_RUNS}; i++ ))
do
    TMP="tmp.lsf"
    echo "#!/bin/bash -x" > ${TMP}
    echo "#BSUB -P med108" >> ${TMP}
    echo "#BSUB -J brca_gen_json_${i}" >> ${TMP}
    echo "#BSUB -o ./logs/log.brca_gen_json.o%J" >> ${TMP}
    echo "#BSUB -e ./logs/log.brca_gen_json.e%J" >> ${TMP}
    echo "#BSUB -W 02:00" >> ${TMP}
    echo "#BSUB -B" >> ${TMP}
    echo "#BSUB -alloc_flags \"smt4\"" >> ${TMP}
    echo "#BSUB -nnodes 1" >> ${TMP}
    echo "source ./0_activate_environment.sh" >> ${TMP}
    echo "source ./0_user_input.sh" >> ${TMP}
    echo "cd \${CODE_DIR}/heatmap_gen" >> ${TMP}
    echo "jsrun -n 1 -a 1 -c 16 -g 0 -b rs ./start_gen_json.sh" >> ${TMP}

    bsub ${TMP}
    rm -f ${TMP}
done
