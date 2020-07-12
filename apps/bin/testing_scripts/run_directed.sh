#!/bin/bash

PROG_NAME=$1
PROG_ARGS=$2

FILE_PREFIX=" -load ./ext_csr_graphs/"
FILE_SUFIX="_ext_CSR.gbin"

PERF_PATTERN="INNER perf"

declare -a file_names=("dir_rmat_20_32"
                       "dir_rmat_21_32"
                       "dir_rmat_22_32"
                       "dir_rmat_23_32"
                       "dir_rmat_24_32"
                       "dir_rmat_25_32"
                       "dir_rmat_26_32"
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
                       "dbpedia"
                       "trackers"
                       "wiki_fr"
                       "wiki_ru"
                       )


rm "$PROG_NAME"_*
rm perf_file.txt
rm full_perf_file.txt

for name in "${file_names[@]}"
do
   CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
   echo "running $CMD_RUN ..."

   eval $CMD_RUN > tmp_file.txt

   search_result=$(grep -R "$PERF_PATTERN" tmp_file.txt)

   echo $search_result
   echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/' >> perf_file.txt
   echo "$name: " >> full_perf_file.txt
   echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/' >> full_perf_file.txt
   echo "---------------" >> full_perf_file.txt
done

rm tmp_file.txt