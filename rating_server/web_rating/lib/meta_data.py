from urllib.request import urlopen
from bs4 import BeautifulSoup


konect_graphs_data = {"soc_pokec": {"link": "soc-pokec-relationships"},
                      "web_baidu": {"link": "zhishi-baidu-internallink"},
                      "road_california": {"link": "roadNet-CA"},
                      "soc_livejournal": {"link": "soc-LiveJournal1"},
                      "road_full_us": {"link": "dimacs9-USA"},
                      "road_central_us": {"link": "dimacs9-CTR"},
                      "web_zhishi": {"link": "zhishi-all"},
                      "web_dbpedia": {"link": "dbpedia-link"},
                      "soc_orkut": {"link": "orkut-links"}}


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

    meta_data_dict = {"graph_category": category}
    return meta_data_dict


def compute_synthetic_metadata(graph_name):
    meta_data_dict = {"graph_category": "synthetic"}
    return meta_data_dict


# kategory types: graph_nature, graph_vertex_scale, graph_edges_scale, graph_edges_factor, graph_max_degree -- should be extendable


def get_meta_data(graph_name):
    if graph_name in konect_graphs_data:
        return load_konect_metadata(konect_graphs_data[graph_name]["link"])
    else:
        return compute_synthetic_metadata(graph_name)


def add_meta_data(received_document, arch):
    document_with_metadata = {}
    document_with_metadata.update(received_document)
    document_with_metadata.update(get_meta_data(received_document["graph_name"]))
    document_with_metadata["arch_name"] = arch
    return document_with_metadata
