from enum import Enum


GENERATE_UNDIRECTED_GRAPHS = False
UNDIRECTED_PREFIX = "undir_"


# "fast" - very fast mode (only small graphs),
# "medium" - medium (small-sized and medium-sized graphs)
# "full" - full (test on all available graphs)
# "one_large" - one large graph per category,
# "rating" - rating mode

print_timings = True


benchmark_args = {"bfs": [ ["-top-down"]],
                  "sssp": [ ["-push", "-all-active"] ],
                  "pr": [ ["-pull"] ],
                  "cc": [ ["-cv"] ],
                  "sswp": [ ["-push"] ],
                  "rw": [ ["-it", "100", "-wv", "20"] ],
                  "hits": [ [] ],
                  "scc": [ [] ],
                  "coloring": [ [] ]}


available_formats = ["csr", "csr_vg", "vcsr", "el", "el_csr_based", "el_2D_seg"]


def requires_undir_graphs(app_name):
    if not GENERATE_UNDIRECTED_GRAPHS:
        return False
    if app_name in ["cc", "coloring"]:
        return True
    return False


common_iterations = 10
perf_pattern = "AVG_PERF"
correctness_pattern = "error count:"

GRAPHS_DIR = "./bin/input_graphs/"
SOURCE_GRAPH_DIR = "./bin/source_graphs/"


# how to add new graph with new category
# 1. add link (http://konect.cc/networks/amazon/ for example) to both dicts
# 2.

all_konect_graphs_data = {
    'soc_catster': {'link': 'petster-friendships-cat'},
    'soc_libimseti': {'link': 'libimseti'},
    'soc_dogster': {'link': 'petster-friendships-dog'},
    'soc_catster_dogster': {'link': 'petster-carnivore'},
    'soc_youtube_friendships': {'link': 'com-youtube'},
    'soc_pokec': {'link': 'soc-pokec-relationships'},
    'soc_orkut': {'link': 'orkut-links'},
    'soc_livejournal': {'link': 'soc-LiveJournal1'},
    'soc_livejournal_links': {'link': 'livejournal-links'},
    'soc_twitter_www': {'link': 'twitter'},
    'soc_friendster': {'link': 'friendster'},

    'web_stanford': {'link': 'web-Stanford'},
    'web_baidu_internal': {'link': 'zhishi-baidu-internallink'},
    'web_wikipedia_links_fr': {'link': 'wikipedia_link_fr'},
    'web_wikipedia_links_ru': {'link': 'wikipedia_link_ru'},
    'web_zhishi': {'link': 'zhishi-all'},
    'web_wikipedia_links_en': {'link': 'wikipedia_link_en'},
    'web_dbpedia_links': {'link': 'dbpedia-link'},
    'web_uk_domain_2002': {'link': 'dimacs10-uk-2002'},
    'web_web_trackers': {'link': 'trackers-trackers'},

    'road_colorado': {'link': 'dimacs9-COL'},
    'road_texas': {'link': 'roadNet-TX'},
    'road_california': {'link': 'roadNet-CA'},
    'road_eastern_usa': {'link': 'dimacs9-E'},
    'road_western_usa': {'link': 'dimacs9-W'},
    'road_central_usa': {'link': 'dimacs9-CTR'},
    'road_full_usa': {'link': 'dimacs9-USA'},

    'rating_yahoo_songs': {'link': 'yahoo-song'},
    'rating_amazon_ratings': {'link': 'amazon-ratings'},
}

#####################

konect_tiny_only = ['soc_libimseti', 'web_stanford', 'road_colorado', 'soc_catster_dogster', 'soc_youtube_friendships',
                    'road_texas', 'rating_yahoo_songs', 'soc_pokec', 'road_california', 'web_baidu_internal']
syn_tiny_only = ["syn_rmat_18_32", "syn_ru_18_32", "syn_rmat_20_32", "syn_ru_20_32"]

#####################

konect_small_only = ['soc_orkut', 'web_wikipedia_links_fr', 'web_wikipedia_links_ru', 'rating_amazon_ratings',
                     'road_eastern_usa', 'soc_livejournal', 'soc_livejournal_links', 'road_western_usa', 'web_zhishi']
syn_small_only = ["syn_rmat_22_32", "syn_ru_22_32"]

#####################

konect_medium_only = ['web_wikipedia_links_en', 'road_central_usa', 'web_dbpedia_links', 'web_uk_domain_2002']
syn_medium_only = ["syn_rmat_24_32", "syn_ru_24_32"]

#####################

konect_large_only = ['road_full_usa', 'web_web_trackers', 'soc_twitter_www', 'soc_friendster']
syn_large_only = ["syn_rmat_25_32", "syn_ru_25_32"]

#####################

konect_tiny_and_small = ['soc_libimseti', 'web_stanford', 'road_colorado', 'soc_catster_dogster',
                         'soc_youtube_friendships', 'road_texas', 'rating_yahoo_songs', 'soc_pokec',
                         'road_california', 'web_baidu_internal',
                         'soc_orkut', 'web_wikipedia_links_fr', 'web_wikipedia_links_ru', 'rating_amazon_ratings',
                         'road_eastern_usa', 'soc_livejournal', 'soc_livejournal_links', 'road_western_usa',
                         'web_zhishi']
syn_tiny_and_small = ["syn_rmat_18_32", "syn_ru_18_32", "syn_rmat_20_32", "syn_ru_20_32",
                      "syn_rmat_22_32", "syn_ru_22_32"]

#####################

konect_rating_full = [
    'web_stanford', 'soc_youtube_friendships', 'road_texas',
    'rating_amazon_ratings', 'soc_livejournal_links', 'road_western_usa', 'web_zhishi',
    'road_central_usa', 'web_uk_domain_2002',
    'road_full_usa', 'web_web_trackers', 'soc_twitter_www'
]
syn_rating_full = ["syn_rmat_18_32", "syn_rmat_22_32", "syn_rmat_24_32", "syn_rmat_25_32",
                   "syn_ru_18_32", "syn_ru_22_32", "syn_ru_24_32", "syn_ru_25_32"]

#####################

konect_rating_fast = [
    'web_stanford', 'soc_youtube_friendships', 'road_texas', #tiny
    'rating_amazon_ratings', 'soc_livejournal_links', 'road_western_usa', 'web_zhishi' #small
]
syn_rating_fast = ["syn_rmat_18_32", "syn_rmat_22_32", "syn_ru_18_32", "syn_ru_22_32"]

#####################

syn_synthetic_only = ["syn_rmat_18_32", "syn_rmat_19_32", "syn_rmat_20_32", "syn_rmat_21_32", "syn_rmat_22_32",
                         "syn_rmat_23_32"]
konect_synthetic_only = []

#####################

apps_and_graphs_ingore = {"sssp": [],
                          "bfs": []}


konect_one_medium_graph_mode = ["web_wiki_ru", "road_western_us", "soc_orkut", "rating_yahoo_songs"]
syn_one_medium_graph_mode = ["syn_rmat_23_32", "syn_ru_23_32"]


