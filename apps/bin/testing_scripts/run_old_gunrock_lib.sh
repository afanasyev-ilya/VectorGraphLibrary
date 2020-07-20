#!/bin/bash

PROG_NAME=$1
PROG_ARGS=$2
PROG_ARGS+=" market "

FILE_PREFIX=" ~/VectorGraphLibrary/apps/bin/gunrock_graphs/undirected/"
FILE_SUFIX="_mtx.el"

PERF_PATTERN="perf:" # can not contain spaces!!!!!!!

declare -a file_names=("undir_rmat_20_32"
                       "undir_rmat_21_32"
                       "undir_rmat_22_32"
                       "undir_rmat_23_32"
                       "undir_rmat_24_32"
                       "undir_rmat_25_32"
                       "undir_ru_20_32"
                       "undir_ru_21_32"
                       "undir_ru_22_32"
                       "undir_ru_23_32"
                       "undir_ru_24_32"
                       "undir_ru_25_32"
                       "friendster"
                       "twitter"
                       "orkut"
                       "lj"
                       "pokec"
                       "wiki_en"
                       "dbpedia"
                       "trackers"
                       "wiki_fr"
                       "wiki_ru"
                       )

rm perf_file.txt
rm full_perf_file.txt

for name in "${file_names[@]}"
do
   CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
   echo "running $CMD_RUN ..."

   eval $CMD_RUN > tmp_file.txt

   echo "$name: " >> full_perf_file.txt
   awk -v PERF_PATTERN="$PERF_PATTERN" '{for (I=1;I<=NF;I++) if ($I == PERF_PATTERN) {print $(I+1)};}' tmp_file.txt >> perf_file.txt
   awk -v PERF_PATTERN="$PERF_PATTERN" '{for (I=1;I<=NF;I++) if ($I == PERF_PATTERN) {print $(I+1)};}' tmp_file.txt >> full_perf_file.txt
   echo "---------------" >> full_perf_file.txt
done

rm tmp_file.txt
