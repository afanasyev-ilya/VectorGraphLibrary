#!/bin/bash

mkdir source_graphs
cd source_graphs

declare -a snap_names=("web-BerkStan.txt.gz"
                       "soc-LiveJournal1.txt.gz"
                       "soc-pokec-relationships.txt.gz"
                       "wiki-Talk.txt.gz"
                       "com-orkut.ungraph.txt.gz"
                       "wiki-topcats.txt.gz"
                       "roadNet-CA.txt.gz"
                       "cit-Patents.txt.gz"
                       "sx-stackoverflow.txt.gz"
                       "as-skitter.txt.gz"
                       "com-friendster.ungraph.txt.gz")

declare -a vgl_names=("web_berk_stan"
                      "soc_lj"
                      "soc_pokec"
                      "wiki_talk"
                      "soc_orkut"
                      "wiki_topcats"
                      "roads_ca"
                      "cit_patents"
                      "soc_stackoverflow"
                      "skitter"
                      "soc_friendster")

for name in "${snap_names[@]}"
do
   CMD="wget https://snap.stanford.edu/data/$name --no-check-certificate"
   echo "running $CMD ..."
   eval $CMD
   CMD="wget https://snap.stanford.edu/data/bigdata/communities/$name --no-check-certificate"
   echo "running $CMD ..."
   eval $CMD
done

for name in "${snap_names[@]}"
do
   CMD="gunzip $name"
   echo "running $CMD ..."
   eval $CMD
done

cd ..

for ((i=0;i<${#snap_names[@]};++i)); do
   SNAP_NAME=${snap_names[i]}
   UNPACKED_NAME=${SNAP_NAME%.gz}
   echo $UNPACKED_NAME
   VGL_NAME=${vgl_names[i]}
   echo $VGL_NAME

   CMD="./create_vgl_graphs_sx -convert ./source_graphs/$UNPACKED_NAME -file ./input_graphs/$VGL_NAME -format vect_csr"
   echo "running $CMD ..."
   eval $CMD
done