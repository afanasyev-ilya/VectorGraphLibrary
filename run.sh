#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -f|--first)
    FIRST="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--last)
    LAST="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--path)
    PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -e|--exec)
    EXEC="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--type)
    GRAPH_TYPE="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "FIRST      = ${FIRST}"
echo "LAST       = ${LAST}"
echo "PATH       = ${PATH}"
echo "EXEC       = ${EXEC}"
echo "GRAPH_TYPE = ${GRAPH_TYPE}"

for ((i=FIRST;i<=LAST;i++));
do
    GRAPH_FILE_NAME="./${PATH}/${GRAPH_TYPE}_${i}_vect_csr.gbin"
    echo "${GRAPH_FILE_NAME}"
    ./cc
done
