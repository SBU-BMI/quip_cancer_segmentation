#!/bin/bash

cd ../
source ./conf/variables.sh

out_folders="heatmap_jsons heatmap_txt json log patch-level-color patch-level-lym patch-level-merged patch-level-nec"
for i in ${out_folders}; do
	if [ ! -d ${OUT_DIR}/$i ]; then
		mkdir -p ${OUT_DIR}/$i
	fi
done
if [ ! -d ${DATA_DIR}/patches ]; then
	mkdir -p ${DATA_DIR}/patches;
fi
wait;

cd patch_extraction_cancer_40X
nohup bash start.sh &
cd ..

cd prediction
nohup bash start.sh &
cd ..

wait;

cd heatmap_gen
nohup bash start.sh &
cd ..

wait;

cd patch_extraction_cancer_40X
nohup bash start.sh &
cd ..

cd prediction
nohup bash start.sh &
cd ..

wait;

cd heatmap_gen
nohup bash start.sh &
cd ..

wait;

exit 0
