#!/bin/bash

check_pattern="Results are equal"

declare -a test_names=("./sssp_sx -s 20 -e 32 -type rmat -push -all-active -check"
                       "./sssp_sx -s 20 -e 32 -type rmat -pull -all-active -check"
                       "./sssp_sx -s 20 -e 32 -type rmat -partial-active -check"
                       "./sssp_sx -s 20 -e 32 -type ru -push -all-active -check"
                       "./sssp_sx -s 20 -e 32 -type ru -pull -all-active -check"
                       "./sssp_sx -s 20 -e 32 -type ru -partial-active -check"

                       "./bfs_sx -s 20 -e 32 -type rmat -td -check"
                       "./bfs_sx -s 20 -e 32 -type ru -td -check"

                       "./scc_sx -s 20 -e 32 -type ru -check"
                       "./scc_sx -s 20 -e 32 -type ru -check")
tests_count=${#test_names[@]}

correct_count=0
error_count=0
for test_name in "${test_names[@]}"
do
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