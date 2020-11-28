#!/bin/bash

SMALLEST=20
LARGEST=25

TYPE="rmat"
for ((SCALE=SMALLEST;SCALE<=LARGEST;SCALE++)); do
   FILE_NAME=$TYPE"_"$SCALE"_32"
   CMD="./create_vgl_graphs_sx -s $SCALE -e 32 -type $TYPE -file ./input_graphs/$FILE_NAME.vgraph"
   echo "running $CMD ..."
   eval $CMD
done

TYPE="ru"
for ((SCALE=SMALLEST;SCALE<=LARGEST;SCALE++)); do
   FILE_NAME=$TYPE"_"$SCALE"_32"
   CMD="./create_vgl_graphs_sx -s $SCALE -e 32 -type $TYPE -file ./input_graphs/$FILE_NAME.vgraph"
   echo "running $CMD ..."
   eval $CMD
done