from urllib.request import urlopen
from bs4 import BeautifulSoup


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


def find_info_on_page(text, pattern):
    for line in text.splitlines():
        if pattern in line:
            return line
    return None


def extract_number(line):
    digits_list = [int(s) for s in line if s.isdigit()]
    return int(''.join(map(str, digits_list)))


def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def extract_category(line):
    return remove_prefix(line, "Category")


#vertices scale tiny
# small
# medium
# large

def get_vertex_scale(num_vertices):
    if num_vertices < pow(2, 16):
        return "tiny_vertex_scale"
    if pow(2, 16) <= num_vertices < pow(2, 22):
        return "small_vertex_scale"
    if pow(2, 22) <= num_vertices < pow(2, 24):
        return "medium_vertex_scale"
    else:
        return "large_vertex_scale"


def load_konect_metadata(graph_link):
    url = "http://konect.cc/networks/" + graph_link
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    page_text = '\n'.join(chunk for chunk in chunks if chunk)
    vertices_count = extract_number(find_info_on_page(page_text, "Size"))
    edges_count = extract_number(find_info_on_page(page_text, "Volume"))
    edges_factor = extract_number(find_info_on_page(page_text, "Average degree"))
    category = extract_category(find_info_on_page(page_text, "Category"))

    #print(vertices_count)
    #print(edges_count)
    #print(edges_factor)
    #print(category)

    meta_data_dict = {"graph_category": category, "vertex_scale": get_vertex_scale(vertices_count)}
    return meta_data_dict


def compute_synthetic_metadata(graph_name):
    vertices_count = pow(2, int(graph_name.split("_")[2]))
    meta_data_dict = {"graph_category": "synthetic", "vertex_scale": get_vertex_scale(vertices_count)}
    return meta_data_dict


# kategory types: graph_nature, graph_vertex_scale, graph_edges_scale, graph_edges_factor, graph_max_degree -- should be extendable


def get_meta_data(graph_name):
    if graph_name in all_konect_graphs_data:
        return load_konect_metadata(all_konect_graphs_data[graph_name]["link"])
    else:
        return compute_synthetic_metadata(graph_name)


def add_meta_data(received_document, arch, arch_dict):
    document_with_metadata = {}
    document_with_metadata.update(received_document)
    document_with_metadata.update(get_meta_data(received_document["graph_name"]))
    document_with_metadata["arch_name"] = arch
    document_with_metadata["arch_dict"] = arch_dict
    return document_with_metadata
