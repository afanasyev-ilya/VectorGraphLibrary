#!/bin/bash

check_pattern="Results are equal"
arch=$1

prefix="sx"

if [[ "$arch" == "gpu" ]]
then
    prefix="cu"
fi

if [[ "$arch" == "nec" ]]
then
    prefix="sx"
fi

declare -a test_names=("./sssp_$prefix -s 20 -e 32 -type rmat -push -all-active -check"
                       "./sssp_$prefix -s 20 -e 32 -type rmat -pull -all-active -check"
                       "./sssp_$prefix -s 20 -e 32 -type rmat -partial-active -check"
                       "./sssp_$prefix -s 20 -e 32 -type ru -push -all-active -check"
                       "./sssp_$prefix -s 20 -e 32 -type ru -pull -all-active -check"
                       "./sssp_$prefix -s 20 -e 32 -type ru -partial-active -check"

                       "./bfs_$prefix -s 20 -e 32 -type rmat -td -check"
                       "./bfs_$prefix -s 20 -e 32 -type ru -td -check"

                       "./scc_$prefix -s 20 -e 32 -type ru -check"
                       "./scc_$prefix -s 20 -e 32 -type ru -check")
tests_count=${#test_names[@]}

correct_count=0
error_count=0
for test_name in "${test_names[@]}"
do
   echo "$test_name"
   cmd_run="$test_name"
   eval $cmd_run > check_file.txt

   results_count=$(grep -c "$check_pattern" check_file.txt)

   if [ $results_count -ge 1 ]
   then
      echo "Test ($test_name) is correct"
      correct_count=$((correct_count+1))
   else
      echo "Error in test ($test_name)"
      error_count=$((error_count+1))
   fi

   rm check_file.txt
done

echo -e "\n\n"
echo "CORRECT: $correct_count / $tests_count"
echo "ERROR: $error_count / $tests_count"