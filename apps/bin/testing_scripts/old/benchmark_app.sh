#!/bin/bash

PROG_NAME=$1
PROG_ARGS=$2

FILE_PREFIX=" -load ./input_graphs/"
FILE_SUFIX=".vgraph"

PERF_PATTERN="MAX_PERF"

declare -a file_names=("rmat_20_32"
                       "rmat_21_32"
                       "rmat_22_32"
                       "rmat_23_32"
                       "rmat_24_32"
                       "rmat_25_32"
                       "ru_20_32"
                       "ru_21_32"
                       "ru_22_32"
                       "ru_23_32"
                       "ru_24_32"
                       "ru_25_32"
                       #"soc_friendster"
                       "soc_stackoverflow"
                       "soc_orkut"
                       "soc_lj"
                       "soc_pokec"
                       "wiki_talk"
                       "wiki_topcats"
                       "berk_stan"
                       "cit_patents"
                       "skitter"
                       "roads_ca"
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