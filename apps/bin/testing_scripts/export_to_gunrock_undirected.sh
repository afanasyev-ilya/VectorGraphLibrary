#!/bin/bash

mkdir gunrock_graphs/undirected

./generate_test_data -push -undirected -s 20 -e 16 -type rmat -file ./gunrock_graphs/undirected/undir_rmat_20_32 -format mtx
./generate_test_data -push -undirected -s 21 -e 16 -type rmat -file ./gunrock_graphs/undirected/undir_rmat_21_32 -format mtx
./generate_test_data -push -undirected -s 22 -e 16 -type rmat -file ./gunrock_graphs/undirected/undir_rmat_22_32 -format mtx
./generate_test_data -push -undirected -s 23 -e 16 -type rmat -file ./gunrock_graphs/undirected/undir_rmat_23_32 -format mtx
./generate_test_data -push -undirected -s 24 -e 16 -type rmat -file ./gunrock_graphs/undirected/undir_rmat_24_32 -format mtx
./generate_test_data -push -undirected -s 25 -e 16 -type rmat -file ./gunrock_graphs/undirected/undir_rmat_25_32 -format mtx

./generate_test_data -push -undirected -s 20 -e 16 -type ru -file ./gunrock_graphs/undirected/undir_ru_20_32 -format mtx
./generate_test_data -push -undirected -s 21 -e 16 -type ru -file ./gunrock_graphs/undirected/undir_ru_21_32 -format mtx
./generate_test_data -push -undirected -s 22 -e 16 -type ru -file ./gunrock_graphs/undirected/undir_ru_22_32 -format mtx
./generate_test_data -push -undirected -s 23 -e 16 -type ru -file ./gunrock_graphs/undirected/undir_ru_23_32 -format mtx
./generate_test_data -push -undirected -s 24 -e 16 -type ru -file ./gunrock_graphs/undirected/undir_ru_24_32 -format mtx
./generate_test_data -push -undirected -s 25 -e 16 -type ru -file ./gunrock_graphs/undirected/undir_ru_25_32 -format mtx

./generate_test_data  -undirected -push -convert ./source_graphs/orkut-links/out.orkut-links -file ./gunrock_graphs/undirected/orkut -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/dbpedia-link/out.dbpedia-link -file ./gunrock_graphs/undirected/dbpedia -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/soc-LiveJournal1/out.soc-LiveJournal1 -file ./gunrock_graphs/undirected/lj -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/soc-pokec-relationships/out.soc-pokec-relationships -file ./gunrock_graphs/undirected/pokec -format mtx

./generate_test_data  -undirected -push -convert ./source_graphs/wikipedia_link_en/out.wikipedia_link_en  -file ./gunrock_graphs/undirected/wiki_en -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/wikipedia_link_fr/out.wikipedia_link_fr -file ./gunrock_graphs/undirected/wiki_fr -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/trackers-trackers/out.trackers -file ./gunrock_graphs/undirected/trackers -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/wikipedia_link_ru/out.wikipedia_link_ru -file ./gunrock_graphs/undirected/wiki_ru -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/twitter/out.twitter  -file ./gunrock_graphs/undirected/twitter -format mtx
./generate_test_data  -undirected -push -convert ./source_graphs/friendster/out.friendster  -file ./gunrock_graphs/undirected/friendster -format mtx

