from .helpers import *
from .common import *


def get_list_of_soc_graphs():
    graphs = []
    if mode == BenchmarkMode.short:
        graphs = ["soc_pokec"]
    elif mode == BenchmarkMode.long:
        graphs = ["soc_stackoverflow",
                  "soc_orkut",
                  "soc_lj",
                  "soc_pokec"]
    return graphs


def get_list_of_misc_graphs():
    graphs = []
    if mode == BenchmarkMode.short:
        graphs = ["wiki_topcats"]
    elif mode == BenchmarkMode.long:
        graphs = ["wiki_talk",
                  "wiki_topcats",
                  "web_berk_stan",
                  "cit_patents",
                  "skitter",
                  "roads_ca"]
    return graphs


def get_list_of_rmat_graphs():
    graphs = []
    edge_factor = synthetic_edge_factor
    min_scale = synthetic_min_scale
    if mode == BenchmarkMode.short:
        max_scale = min_scale + 2
    elif mode == BenchmarkMode.long:
        max_scale = synthetic_max_scale
    for scale in range(min_scale, max_scale + 1):
        new_graph = "rmat_" + str(scale) + "_" + str(edge_factor)
        graphs += [new_graph]
    return graphs


def get_list_of_ru_graphs():
    graphs = []
    edge_factor = synthetic_edge_factor
    min_scale = synthetic_min_scale
    if mode == BenchmarkMode.short:
        max_scale = min_scale + 2
    elif mode == BenchmarkMode.long:
        max_scale = synthetic_max_scale
    for scale in range(min_scale, max_scale + 1):
        new_graph = "ru_" + str(scale) + "_" + str(edge_factor)
        graphs += [new_graph]
    return graphs


def get_list_of_verification_graphs():
    return ["rmat_20_"+str(synthetic_edge_factor),
            "ru_20_"+str(synthetic_edge_factor),
            "soc_pokec",
            "wiki_talk"]


def get_list_of_all_graphs():
    list_of_graphs = get_list_of_rmat_graphs() + get_list_of_ru_graphs() + get_list_of_soc_graphs() + \
                     get_list_of_misc_graphs() + get_list_of_verification_graphs()
    return list_of_graphs


def get_graph_path(graph_name):
    prefix = GRAPHS_DIR
    suffix =".vgraph"
    return prefix + graph_name + suffix


def create_real_world_graph(graph_name, arch):
    snap_links = {"web_berk_stan": "web-BerkStan.txt.gz",
                  "soc_lj":  "soc-LiveJournal1.txt.gz",
                  "soc_pokec": "soc-pokec-relationships.txt.gz",
                  "wiki_talk": "wiki-topcats.txt.gz",
                  "soc_orkut": "com-orkut.ungraph.txt.gz",
                  "wiki_topcats": "wiki-topcats.txt.gz",
                  "roads_ca": "roadNet-CA.txt.gz",
                  "cit_patents": "cit-Patents.txt.gz",
                  "soc_stackoverflow": "sx-stackoverflow.txt.gz",
                  "skitter": "as-skitter.txt.gz"}
    link = "https://snap.stanford.edu/data/" + snap_links[graph_name]
    cmd = ["wget", link, "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
    print(cmd)
    subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

    link = "https://snap.stanford.edu/data/bigdata/communities/" + snap_links[graph_name]
    cmd = ["wget", link, "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
    print(cmd)
    subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

    tar_name = SOURCE_GRAPH_DIR + snap_links[graph_name]
    cmd = ["gunzip", "-f", tar_name]
    subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

    source_name = os.path.splitext(tar_name)[0]
    cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-directed", "-file", GRAPHS_DIR + graph_name]
    print(cmd)
    subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()


def create_synthetic_graph(graph_name, arch):
    dat = graph_name.split("_")
    type = dat[0]
    scale = dat[1]
    edge_factor = dat[2]

    cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor, "-type", type, "-directed", "-file", GRAPHS_DIR + graph_name]
    print(cmd)
    subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()


def create_graph(graph_name, arch):
    if "rmat" in graph_name or "ru" in graph_name:
        create_synthetic_graph(graph_name, arch)
    else:
        create_real_world_graph(graph_name, arch)


def create_graphs_if_required(list_of_graphs, arch):
    create_dir(GRAPHS_DIR)
    create_dir(SOURCE_GRAPH_DIR)

    if not binary_exists("create_vgl_graphs", arch):
        make_binary("create_vgl_graphs", arch)

    for current_graph in list_of_graphs:
        if not file_exists(get_graph_path(current_graph)):
            print("Warning! need to create graph " + get_graph_path(current_graph))
            create_graph(current_graph, arch)


