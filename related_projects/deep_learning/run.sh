#!/bin/bash

python3 ./prepare_all_data.py

for i in {0..5}; do
   cd $i
   ls -1 | wc -l
   cd ..
done