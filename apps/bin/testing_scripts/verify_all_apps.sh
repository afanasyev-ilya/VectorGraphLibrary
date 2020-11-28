#!/bin/bash

FILE_PREFIX=" -load ./input_graphs/"
FILE_SUFIX=".vgraph"
CHECK_PATTERN="error count: "
file_name="./nec_check.csv"
declare -A matrix

declare -a graph_names=("ru_20_32"
                        "rmat_20_32"
                        "soc_pokec"
                        "wiki_topcats")

declare -a app_names=("./bfs_sx -td "
                      "./sssp_sx -pull -all-active "
                      "./sssp_sx -push -all-active ")

declare -a app_column_names=("./bfs_sx|-td"
                         "./sssp_sx|-pull|-all-active"
                         "./sssp_sx|-push|-all-active"
                         "./sssp_sx|-partial-active"
                         "./cc_sx"
                         "./pr_sx")

num_rows=5
num_columns=7

for ((row=1;row<=num_rows;row++)) do
    for ((col=1;col<=num_columns;col++)) do
        matrix[$row,$col]=""
    done
done

it="2"
for name in "${app_column_names[@]}"
do
    matrix["1",$it]=$name
    it=$((it+1))
done

it="2"
for name in "${graph_names[@]}"
do
    matrix[$it,"1"]=$name
    it=$((it+1))
done

rm $file_name

correct_tests="0"
error_tests="0"
all_tests="0"
col="2"
for app in "${app_names[@]}"
do
    row="2"
    for graph_name in "${graph_names[@]}"
    do
        CMD_RUN="$app $FILE_PREFIX$graph_name$FILE_SUFIX -check"
        #echo "running $CMD_RUN ..."
        eval $CMD_RUN > tmp_file.txt
        search_result=$(grep -R "$CHECK_PATTERN" tmp_file.txt)
        error_count=`echo $search_result | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/'`
        #echo "$error_count"

        if [ "$error_count" = "0" ]; then
            #echo "Correct!"
            correct_tests=$((correct_tests+1))
            all_tests=$((all_tests+1))
            matrix[$row,$col]="correct"
        else
            #echo "Errors in $app test: $error_count"
            error_tests=$((error_tests+1))
            all_tests=$((all_tests+1))
            matrix[$row,$col]="errors:"$error_count
        fi
        row=$((row+1))
    done
    col=$((col+1))
done

rm tmp_file.txt

echo "Correct : $correct_tests/$all_tests"
echo "Errors : $error_tests/$all_tests"

for ((row=1;row<=num_rows;row++)) do
    for ((col=1;col<=num_columns;col++)) do
        printf ${matrix[$row,$col]}"," >> $file_name
    done
    printf " " >> $file_name
    printf "\n" >> $file_name
done