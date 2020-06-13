#!/bin/bash

mkdir gunrock_graphs

./generate_test_data -push -directed -s 20 -e 32 -type rmat -file ./gunrock_graphs/dir_rmat_20_32 -format mtx
./generate_test_data -push -directed -s 21 -e 32 -type rmat -file ./gunrock_graphs/dir_rmat_21_32 -format mtx
./generate_test_data -push -directed -s 22 -e 32 -type rmat -file ./gunrock_graphs/dir_rmat_22_32 -format mtx
./generate_test_data -push -directed -s 23 -e 32 -type rmat -file ./gunrock_graphs/dir_rmat_23_32 -format mtx
./generate_test_data -push -directed -s 24 -e 32 -type rmat -file ./gunrock_graphs/dir_rmat_24_32 -format mtx
./generate_test_data -push -directed -s 25 -e 32 -type rmat -file ./gunrock_graphs/dir_rmat_25_32 -format mtx

./generate_test_data -push -directed -s 20 -e 32 -type ru -file ./gunrock_graphs/ru_20_32 -format mtx
./generate_test_data -push -directed -s 21 -e 32 -type ru -file ./gunrock_graphs/ru_21_32 -format mtx
./generate_test_data -push -directed -s 22 -e 32 -type ru -file ./gunrock_graphs/ru_22_32 -format mtx
./generate_test_data -push -directed -s 23 -e 32 -type ru -file ./gunrock_graphs/ru_23_32 -format mtx
./generate_test_data -push -directed -s 24 -e 32 -type ru -file ./gunrock_graphs/ru_24_32 -format mtx
./generate_test_data -push -directed -s 25 -e 32 -type ru -file ./gunrock_graphs/ru_25_32 -format mtx

#./generate_test_data  -directed -push -convert ./source_graphs/orkut-links/out.orkut-links -file ./gunrock_graphs/orkut -format mtx
#./generate_test_data  -directed -push -convert ./source_graphs/dbpedia-link/out.dbpedia-link -file ./gunrock_graphs/dbpedia -format mtx
#./generate_test_data  -directed -push -convert ./source_graphs/soc-LiveJournal1/out.soc-LiveJournal1 -file ./gunrock_graphs/lj -format mtx
#./generate_test_data  -directed -push -convert ./source_graphs/soc-pokec-relationships/out.soc-pokec-relationships -file ./gunrock_graphs/pokec -format mtx

#./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_en/out.wikipedia_link_en  -file ./gunrock_graphs/wiki_en -format mtx
./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_fr/out.wikipedia_link_fr -file ./gunrock_graphs/wiki_fr -format mtx
#./generate_test_data  -directed -push -convert ./source_graphs/trackers-trackers/out.trackers -file ./gunrock_graphs/trackers -format mtx
#./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_ru/out.wikipedia_link_ru -file ./gunrock_graphs/wiki_ru -format mtx
#./generate_test_data  -directed -push -convert ./source_graphs/twitter/out.twitter  -file ./gunrock_graphs/twitter -format mtx

