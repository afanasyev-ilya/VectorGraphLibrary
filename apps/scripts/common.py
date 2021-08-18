from enum import Enum


GENERATE_UNDIRECTED_GRAPHS = False
UNDIRECTED_PREFIX = "undir_"


benchmark_args = {"bfs": [ ["-top-down"]],
                  "sssp": [ ["-push", "-all-active"] ],
                  "pr": [ ["-pull"] ],
                  "cc": [ ["-cv"] ],
                  "sswp": [ ["-push"] ],
                  "rw": [ ["-it", "100", "-wv", "20"] ],
                  "hits": [ [] ],
                  "scc": [ [] ],
                  "coloring": [ [] ]}

available_formats = ["csr", "vcsr"]


def requires_undir_graphs(app_name):
    if not GENERATE_UNDIRECTED_GRAPHS:
        return False
    if app_name in ["cc", "coloring"]:
        return True
    return False


common_iterations = 10
perf_pattern = "MAX_PERF"
correctness_pattern = "error count:"

GRAPHS_DIR = "./bin/input_graphs/"
SOURCE_GRAPH_DIR = "./bin/source_graphs/"


# how to add new graph with new category
# 1. add link (http://konect.cc/networks/amazon/ for example) to both dicts
# 2.


konect_graphs_data_fast = {"soc_pokec": {"link": "soc-pokec-relationships"},
                      "soc_livejournal": {"link": "soc-LiveJournal1"},
                      #
                      "web_wikipedia_link_en": {"link": "wikipedia_link_en"},
                      "web_baidu": {"link": "zhishi-baidu-internallink"},
                      #
                      "road_california": {"link": "roadNet-CA"},
                      "road_colorado": {"link": "dimacs9-COL"},
                      #
                      "rating_yahoo_song": {"link": "yahoo-song"},
                      "rating_amazon_ratings": {"link": "amazon-ratings"}}


konect_graphs_data = {"soc_pokec": {"link": "soc-pokec-relationships"},
                      "soc_livejournal": {"link": "soc-LiveJournal1"},
                      "soc_orkut": {"link": "orkut-links"},
                      "soc_catster_dogster": {"link": "petster-carnivore"},
                      "soc_libimseti": {"link": "libimseti"}, # 157 degree
                      "soc_youtube_friendship": {"link": "youtube-u-growth"}, # low friends - 5
                      #
                      "web_wikipedia_link_en": {"link": "wikipedia_link_en"},
                      "web_baidu": {"link": "zhishi-baidu-internallink"},
                      "web_zhishi": {"link": "zhishi-all"},
                      "web_dbpedia": {"link": "dbpedia-link"},
                      "web_trackers": {"link": "trackers-trackers", "unarch_graph_name": "trackers"},
                      #
                      "road_full_us": {"link": "dimacs9-USA"},
                      "road_central_us": {"link": "dimacs9-CTR"},
                      "road_california": {"link": "roadNet-CA"},
                      "road_eastern_us": {"link": "dimacs9-E"},
                      "road_western_us": {"link": "dimacs9-W"},
                      "road_colorado": {"link": "dimacs9-COL"},
                      #
                      "rating_yahoo_song": {"link": "yahoo-song"},
                      "rating_amazon_ratings": {"link": "amazon-ratings"}}

apps_and_graphs_ingore = {"sssp": [],
                          "bfs": []}


#snap_links = {"web_berk_stan": "web-BerkStan.txt.gz",
#              "soc_lj":  "soc-LiveJournal1.txt.gz",
#              "soc_pokec": "soc-pokec-relationships.txt.gz",
#              "wiki_talk": "wiki-topcats.txt.gz",
#              "soc_orkut": "com-orkut.ungraph.txt.gz",
#              "wiki_topcats": "wiki-topcats.txt.gz",
#              "roads_ca": "roadNet-CA.txt.gz",
#              "cit_patents": "cit-Patents.txt.gz",
#              "soc_stackoverflow": "sx-stackoverflow.txt.gz",
#              "skitter": "as-skitter.txt.gz"}

