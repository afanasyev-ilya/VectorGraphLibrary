#!/bin/bash

arch="sx"

if [ $# -eq 1 ]
  then
    arch=$1
fi

file_name="./"$arch"_check.csv"
rm $file_name

file_prefix=" -load ./input_graphs/"
file_sufix=".vgraph"
check_pattern="error count: "

declare -A matrix

declare -a graph_names=("ru_20_32"
                        "rmat_20_32"
                        "soc_pokec"
                        "wiki_topcats")

declare -a app_names=("./bfs_"$arch" -td "
                      "./sssp_"$arch" -pull -all-active "
                      "./sssp_"$arch" -push -all-active "
                      "./cc_"$arch" -shiloach_vishkin",
                      "./pr_"$arch" -it 5")

declare -a app_column_names=("./bfs_"$arch"|-td"
                             "./sssp_"$arch"|-pull|-all-active"
                             "./sssp_"$arch"|-push|-all-active"
                             "./cc_"$arch"|-shiloach_vishkin",
                             "./pr"$arch"|-it|5")

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
        CMD_RUN="$app $file_prefix$graph_name$file_sufix -check"
        #echo "running $CMD_RUN ..."
        eval $CMD_RUN > tmp_file.txt
        search_result=$(grep -R "$check_pattern" tmp_file.txt)
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