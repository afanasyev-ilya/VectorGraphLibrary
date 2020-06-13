#!/bin/bash

PROG_NAME="./pr "
PROG_ARGS="--remove-self-loops=false --remove-duplicate-edges=false --graph-type market --graph-file "

FILE_PREFIX=" ~/VectorGraphLibrary/apps/bin/gunrock_graphs/"
FILE_SUFIX="_mtx.el"

PERF_PATTERN="perf:"

declare -a file_names=("dir_rmat_20_32"
                       "dir_rmat_21_32"
                       "dir_rmat_22_32"
                       "dir_rmat_23_32"
                       "dir_rmat_24_32"
                       "dir_rmat_25_32"
                       "dir_ru_20_32"
                       "dir_ru_21_32"
                       "dir_ru_22_32"
                       "dir_ru_23_32"
                       "dir_ru_24_32"
                       "dir_ru_25_32"
                       "friendster"
                       "twitter"
                       "orkut"
                       "lj"
                       "pokec"
                       "wiki_en"
                       "trackers"
                       "wiki_fr"
                       "wiki_fr"
                       "wiki_ru"
                       )

rm perf_file.txt
rm full_perf_file.txt

for name in "${file_names[@]}"
do
   CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
   echo "running $CMD_RUN ..."

   eval $CMD_RUN

   #eval $CMD_RUN > tmp_file.txt

   #echo "$name: " >> full_perf_file.txt
   #awk '{for (I=1;I<=NF;I++) if ($I == "rate:") {print $(I+1)};}' tmp_file.txt >> perf_file.txt
   #awk '{for (I=1;I<=NF;I++) if ($I == "rate:") {print $(I+1)};}' tmp_file.txt >> full_perf_file.txt
   #echo "---------------" >> full_perf_file.txt

   # or do whatever with individual element of the array
done

rm tmp_file.txt
