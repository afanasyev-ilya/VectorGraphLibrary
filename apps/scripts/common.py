from enum import Enum


GENERATE_UNDIRECTED_GRAPHS = False
UNDIRECTED_PREFIX = "undir_"


benchmark_args = {"bfs": [ ["-top-down"], ["-do"] ],
                  "sssp": [ ["-push", "-all-active"], ["-pull", "-all-active"], ["-push", "-partial-active"] ],
                  "pr": [ ["-pull"] ],
                  "cc": [ ["-bfs-based"], ["-cv"] ],
                  "sswp": [ ["-push"] ],
                  "rw": [ ["-it", "100", "-wv", "20"] ],
                  "hits": [ [] ],
                  "scc": [ [] ],
                  "coloring": [ [] ]}


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
synthetic_min_scale = 18
synthetic_medium_scale = 24
synthetic_large_scale = 26
synthetic_edge_factor = 32


class BenchmarkMode(Enum):
    long = 1,
    medium = 2
    short = 3


mode = BenchmarkMode.medium


konect_links = {"soc_pokec": "soc-pokec-relationships",
                "web_baidu": "zhishi-baidu-internallink",
                "road_california": "roadNet-CA"}


#http://konect.cc/files/download.tsv.zhishi-baidu-internallink.tar.bz2
#http://konect.cc/files/download.tsv.soc-pokec-relationships.tar.bz2
#http://konect.cc/files/download.tsv.trackers-trackers.tar.bz2
#http://konect.cc/files/download.tsv.roadNet-CA.tar.bz2


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

