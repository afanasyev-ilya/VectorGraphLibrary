#!/bin/bash

mkdir ligra_graphs

PATH_TO_LIGRA=/home/mq.cal.is.tohoku.ac.jp/afanasiev/cpu_graph_frameworks/ligra/utils

./generate_test_data -push -directed -s 20 -e 32 -type rmat -file ./ligra_graphs/dir_rmat_20_32 -format ligra
./generate_test_data -push -directed -s 21 -e 32 -type rmat -file ./ligra_graphs/dir_rmat_21_32 -format ligra
./generate_test_data -push -directed -s 22 -e 32 -type rmat -file ./ligra_graphs/dir_rmat_22_32 -format ligra
./generate_test_data -push -directed -s 23 -e 32 -type rmat -file ./ligra_graphs/dir_rmat_23_32 -format ligra
./generate_test_data -push -directed -s 24 -e 32 -type rmat -file ./ligra_graphs/dir_rmat_24_32 -format ligra
./generate_test_data -push -directed -s 25 -e 32 -type rmat -file ./ligra_graphs/dir_rmat_25_32 -format ligra

./generate_test_data -push -directed -s 20 -e 32 -type ru -file ./ligra_graphs/dir_ru_20_32 -format ligra
./generate_test_data -push -directed -s 21 -e 32 -type ru -file ./ligra_graphs/dir_ru_21_32 -format ligra
./generate_test_data -push -directed -s 22 -e 32 -type ru -file ./ligra_graphs/dir_ru_22_32 -format ligra
./generate_test_data -push -directed -s 23 -e 32 -type ru -file ./ligra_graphs/dir_ru_23_32 -format ligra
./generate_test_data -push -directed -s 24 -e 32 -type ru -file ./ligra_graphs/dir_ru_24_32 -format ligra
./generate_test_data -push -directed -s 25 -e 32 -type ru -file ./ligra_graphs/dir_ru_25_32 -format ligra

./generate_test_data  -directed -push -convert ./source_graphs/orkut-links/out.orkut-links -file ./ligra_graphs/orkut -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/dbpedia-link/out.dbpedia-link -file ./ligra_graphs/dbpedia -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/soc-LiveJournal1/out.soc-LiveJournal1 -file ./ligra_graphs/lj -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/soc-pokec-relationships/out.soc-pokec-relationships -file ./ligra_graphs/pokec -format ligra

./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_en/out.wikipedia_link_en  -file ./ligra_graphs/wiki_en -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_fr/out.wikipedia_link_fr -file ./ligra_graphs/wiki_fr -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/trackers-trackers/out.trackers -file ./ligra_graphs/trackers -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_ru/out.wikipedia_link_ru -file ./ligra_graphs/wiki_ru -format ligra
./generate_test_data  -directed -push -convert ./source_graphs/twitter/out.twitter  -file ./ligra_graphs/twitter -format ligra


$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_rmat_20_32_ligra.txt ./ligra_graphs/wdir_rmat_20_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_rmat_21_32_ligra.txt ./ligra_graphs/wdir_rmat_21_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_rmat_22_32_ligra.txt ./ligra_graphs/wdir_rmat_22_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_rmat_23_32_ligra.txt ./ligra_graphs/wdir_rmat_23_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_rmat_24_32_ligra.txt ./ligra_graphs/wdir_rmat_24_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_rmat_25_32_ligra.txt ./ligra_graphs/wdir_rmat_25_32_ligra.txt


$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_ru_20_32_ligra.txt ./ligra_graphs/wdir_ru_20_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_ru_21_32_ligra.txt ./ligra_graphs/wdir_ru_21_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_ru_22_32_ligra.txt ./ligra_graphs/wdir_ru_22_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_ru_23_32_ligra.txt ./ligra_graphs/wdir_ru_23_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_ru_24_32_ligra.txt ./ligra_graphs/wdir_ru_24_32_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dir_ru_25_32_ligra.txt ./ligra_graphs/wdir_ru_25_32_ligra.txt

$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/orkut_ligra.txt ./ligra_graphs/workut_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/dbpedia_ligra.txt ./ligra_graphs/wdbpedia_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/lj_ligra.txt ./ligra_graphs/wlj_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/pokec_ligra.txt ./ligra_graphs/wpokec_ligra.txt

$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/wiki_en_ligra.txt ./ligra_graphs/wwiki_en_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/wiki_fr_ligra.txt ./ligra_graphs/wwiki_fr_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/wiki_ru_ligra.txt ./ligra_graphs/wwiki_ru_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/trackers_ligra.txt ./ligra_graphs/wtrackers_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/twitter_ligra.txt ./ligra_graphs/wtwitter_ligra.txt
$PATH_TO_LIGRA/adjGraphAddWeights ./ligra_graphs/friendster_ligra.txt ./ligra_graphs/wfriendster_ligra.txt



