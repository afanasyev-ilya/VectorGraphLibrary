#!/bin/bash

mkdir gunrock_graphs

./generate_test_data -directed -push -convert ./source_graphs/orkut-links/out.orkut-links -file ./gunrock_graphs/orkut -format mtx
