#!/bin/bash

mkdir source_graphs
cd source_graphs

wget http://konect.uni-koblenz.de/downloads/tsv/orkut-links.tar.bz2
wget http://konect.uni-koblenz.de/downloads/tsv/soc-LiveJournal1.tar.bz2
wget http://konect.uni-koblenz.de/downloads/tsv/soc-pokec-relationships.tar.bz2

wget http://konect.uni-koblenz.de/downloads/tsv/wikipedia_link_en.tar.bz2
wget http://konect.uni-koblenz.de/downloads/tsv/dbpedia-link.tar.bz2
wget http://konect.uni-koblenz.de/downloads/tsv/trackers-trackers.tar.bz2
wget http://konect.uni-koblenz.de/downloads/tsv/wikipedia_link_fr.tar.bz
wget http://konect.uni-koblenz.de/downloads/tsv/wikipedia_link_ru.tar.bz2

wget http://konect.uni-koblenz.de/downloads/tsv/friendster.tar.bz2
wget http://konect.uni-koblenz.de/downloads/tsv/twitter.tar.bz2

tar -xvf dbpedia-link.tar.bz2
tar -xvf soc-LiveJournal1.tar.bz2
tar -xvf wikipedia_link_ru.tar.bz2
tar -xvf soc-pokec-relationships.tar.bz2
tar -xvf wikipedia_link_en.tar.bz2
tar -xvf orkut-links.tar.bz2
tar -xvf trackers-trackers.tar.bz2
tar -xvf wikipedia_link_fr.tar.bz

tar -xvf friendster.tar.bz2
tar -xvf twitter.tar.bz2

cd ..