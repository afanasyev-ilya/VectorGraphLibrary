!/bin/bash

mkdir gunrock_graphs

FOLDER="./galois_graphs"
DIRECTION="-directed"
FORMAT="-galois"

mkdir $FOLDER

./generate_test_data -push $DIRECTION -s 20 -e 32 -type rmat -file $FOLDER/dir_rmat_20_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 21 -e 32 -type rmat -file $FOLDER/dir_rmat_21_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 22 -e 32 -type rmat -file $FOLDER/dir_rmat_22_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 23 -e 32 -type rmat -file $FOLDER/dir_rmat_23_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 24 -e 32 -type rmat -file $FOLDER/dir_rmat_24_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 25 -e 32 -type rmat -file $FOLDER/dir_rmat_25_32 -format $FORMAT

./generate_test_data -push $DIRECTION -s 20 -e 32 -type ru -file $FOLDER/dir_ru_20_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 21 -e 32 -type ru -file $FOLDER/dir_ru_21_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 22 -e 32 -type ru -file $FOLDER/dir_ru_22_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 23 -e 32 -type ru -file $FOLDER/dir_ru_23_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 24 -e 32 -type ru -file $FOLDER/dir_ru_24_32 -format $FORMAT
./generate_test_data -push $DIRECTION -s 25 -e 32 -type ru -file $FOLDER/dir_ru_25_32 -format $FORMAT

./generate_test_data  $DIRECTION -push -convert ./source_graphs/orkut-links/out.orkut-links -file $FOLDER/orkut -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/dbpedia-link/out.dbpedia-link -file $FOLDER/dbpedia -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/soc-LiveJournal1/out.soc-LiveJournal1 -file $FOLDER/lj -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/soc-pokec-relationships/out.soc-pokec-relationships -file $FOLDER/pokec -format $FORMAT

./generate_test_data  $DIRECTION -push -convert ./source_graphs/wikipedia_link_en/out.wikipedia_link_en  -file $FOLDER/wiki_en -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/wikipedia_link_fr/out.wikipedia_link_fr -file $FOLDER/wiki_fr -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/trackers-trackers/out.trackers -file $FOLDER/trackers -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/wikipedia_link_ru/out.wikipedia_link_ru -file $FOLDER/wiki_ru -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/twitter/out.twitter  -file $FOLDER/twitter -format $FORMAT
./generate_test_data  $DIRECTION -push -convert ./source_graphs/friendster/out.friendster  -file $FOLDER/friendster -format $FORMAT

