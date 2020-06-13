#!/bin/bash

./generate_test_data  -directed -push -convert ./source_graphs/orkut-links/out.orkut-links -file ./ext_csr_graphs/orkut
./generate_test_data  -directed -push -convert ./source_graphs/dbpedia-link/out.dbpedia-link -file ./ext_csr_graphs/dbpedia
./generate_test_data  -directed -push -convert ./source_graphs/soc-LiveJournal1/out.soc-LiveJournal1 -file ./ext_csr_graphs/lj
./generate_test_data  -directed -push -convert ./source_graphs/soc-pokec-relationships/out.soc-pokec-relationships  -file ./ext_csr_graphs/pokec

./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_en/out.wikipedia_link_en  -file ./ext_csr_graphs/wiki_en
./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_fr/out.wikipedia_link_fr -file ./ext_csr_graphs/wiki_fr
./generate_test_data  -directed -push -convert ./source_graphs/trackers-trackers/out.trackers -file ./ext_csr_graphs/trackers
./generate_test_data  -directed -push -convert ./source_graphs/wikipedia_link_ru/out.wikipedia_link_ru -file ./ext_csr_graphs/wiki_ru

./generate_test_data  -directed -push -convert ./source_graphs/twitter/out.twitter  -file ./ext_csr_graphs/twitter