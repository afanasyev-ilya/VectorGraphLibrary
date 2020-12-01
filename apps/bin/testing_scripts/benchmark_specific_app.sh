#!/bin/bash

declare -A matrix
PROG_NAME=$1
PROG_ARGS=$2
arch=$3
file_name="./"$arch"_performance_stats.csv"
algorithm_name=$PROG_NAME" "$PROG_ARGS
algorithm_name=$(sed 's/ /|/g' <<< "$algorithm_name")

FILE_PREFIX=" -load ./input_graphs/"
FILE_SUFIX=".vgraph"

PERF_PATTERN="MAX_PERF"

num_rows=8
num_columns=10

for ((row=1;row<=num_rows;row++)) do
    for ((col=1;col<=num_columns;col++)) do
        matrix[$row,$col]=""
    done
done

declare -a column_names=("algorithm_name"
                         "rmat_graph"
                         "rmat_perf"
                         "ru_graph"
                         "ru_perf"
                         "soc_graph"
                         "soc_perf"
                         "misc_graph"
                         "misc_perf")

it="1"
for name in "${column_names[@]}"
do
    matrix["1",$it]=$name
    it=$((it+1))
done

matrix["2","1"]=$algorithm_name
for ((row=3;row<=num_rows;row++))
do
    matrix[$row,"1"]=""
done

declare -a rmat_names=("rmat_20_32"
                       "rmat_21_32"
                       "rmat_22_32"
                       "rmat_23_32"
                       "rmat_24_32"
                       "rmat_25_32")

it="2"
for name in "${rmat_names[@]}"
do
    matrix[$it,"2"]=$name
    it=$((it+1))
done

it="2"
for name in "${rmat_names[@]}"
do
    CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
    echo "running $CMD_RUN ..."
    eval $CMD_RUN > tmp_file.txt
    search_result=$(grep -R "$PERF_PATTERN" tmp_file.txt)
    perf=`echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/'`
    echo "$perf"
    matrix[$it,"3"]=$perf
    it=$((it+1))
done

declare -a ru_names=("ru_20_32"
                     "ru_21_32"
                     "ru_22_32"
                     "ru_23_32"
                     "ru_24_32"
                     "ru_25_32")

it="2"
for name in "${ru_names[@]}"
do
    matrix[$it,"4"]=$name
    it=$((it+1))
done

it="2"
for name in "${ru_names[@]}"
do
    CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
    echo "running $CMD_RUN ..."
    eval $CMD_RUN > tmp_file.txt
    search_result=$(grep -R "$PERF_PATTERN" tmp_file.txt)
    perf=`echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/'`
    echo "$perf"
    matrix[$it,"5"]=$perf
    it=$((it+1))
done

declare -a soc_names=("soc_stackoverflow"
                       "soc_orkut"
                       "soc_lj"
                       "soc_pokec")

it="2"
for name in "${soc_names[@]}"
do
    matrix[$it,"6"]=$name
    it=$((it+1))
done

it="2"
for name in "${soc_names[@]}"
do
    CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
    echo "running $CMD_RUN ..."
    eval $CMD_RUN > tmp_file.txt
    search_result=$(grep -R "$PERF_PATTERN" tmp_file.txt)
    perf=`echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/'`
    echo "$perf"
    matrix[$it,"7"]=$perf
    it=$((it+1))
done

declare -a misc_names=("wiki_talk"
                       "wiki_topcats"
                       "berk_stan"
                       "cit_patents"
                       "skitter"
                       "roads_ca")

it="2"
for name in "${misc_names[@]}"
do
    matrix[$it,"8"]=$name
    it=$((it+1))
done

it="2"
for name in "${misc_names[@]}"
do
    CMD_RUN="$PROG_NAME $PROG_ARGS$FILE_PREFIX$name$FILE_SUFIX "
    echo "running $CMD_RUN ..."
    eval $CMD_RUN > tmp_file.txt
    search_result=$(grep -R "$PERF_PATTERN" tmp_file.txt)
    perf=`echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/'`
    echo "$perf"
    matrix[$it,"9"]=$perf
    it=$((it+1))
done

for ((row=1;row<=num_rows;row++)) do
    for ((col=1;col<=num_columns;col++)) do
        printf ${matrix[$row,$col]}"," >> $file_name
    done
    printf " " >> $file_name
    printf "\n" >> $file_name
done

rm tmp_file.txt