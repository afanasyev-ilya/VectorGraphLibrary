from .helpers import *
from .common import *
import os.path
from os import path


def get_list_of_soc_graphs():
    graphs = []
    if mode == BenchmarkMode.short:
        graphs = ["soc_pokec"]
    elif (mode == BenchmarkMode.long) or (mode == BenchmarkMode.medium):
        graphs = ["soc_stackoverflow",
                  "soc_orkut",
                  "soc_lj",
                  "soc_pokec"]
    return graphs


def get_list_of_misc_graphs():
    graphs = []
    if mode == BenchmarkMode.short:
        graphs = ["wiki_topcats"]
    elif (mode == BenchmarkMode.long) or (mode == BenchmarkMode.medium):
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
    max_scale = min_scale + 1
    if mode == BenchmarkMode.short:
        max_scale = min_scale + 2
    elif mode == BenchmarkMode.medium:
        max_scale = synthetic_medium_scale
    elif mode == BenchmarkMode.long:
        max_scale = synthetic_medium_scale
    for scale in range(min_scale, max_scale + 1):
        new_graph = "rmat_" + str(scale) + "_" + str(edge_factor)
        graphs += [new_graph]
    return graphs


def get_list_of_ru_graphs():
    graphs = []
    edge_factor = synthetic_edge_factor
    min_scale = synthetic_min_scale
    max_scale = min_scale + 1
    if mode == BenchmarkMode.short:
        max_scale = min_scale + 2
    elif mode == BenchmarkMode.medium:
        max_scale = synthetic_medium_scale
    elif mode == BenchmarkMode.long:
        max_scale = synthetic_medium_scale
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


def download_snap_graphs():
    snap_graphs = get_list_of_soc_graphs() + get_list_of_misc_graphs()
    for graph_name in snap_graphs:
        download_graph(graph_name)


def download_graph(graph_name):
    if not path.exists(SOURCE_GRAPH_DIR + "/" + snap_links[graph_name]):
        print("Trying to download " + SOURCE_GRAPH_DIR + snap_links[graph_name])

        if not internet_on():
            print("Error! no internet connection available.")
            return

        link = "https://snap.stanford.edu/data/" + snap_links[graph_name]
        cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        link = "https://snap.stanford.edu/data/bigdata/communities/" + snap_links[graph_name]
        cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

    if path.exists(SOURCE_GRAPH_DIR + snap_links[graph_name]):
        print("File " + SOURCE_GRAPH_DIR + snap_links[graph_name] + " downloaded!")
    else:
        print("Error! Can not download file " + SOURCE_GRAPH_DIR + snap_links[graph_name])


def get_path_to_graph(short_name, undir = False):
    if undir:
        return GRAPHS_DIR + UNDIRECTED_PREFIX + short_name + ".vgraph"
    else:
        return GRAPHS_DIR + short_name + ".vgraph"


def verify_graph_existence(graph_file_name):
    if file_exists(graph_file_name):
        print("Success! graph " + graph_file_name + " has been created.")
        return True
    else:
        print("Error! graph " + graph_file_name + " can not be created.")
        return False


def create_real_world_graph(graph_name, arch):
    output_graph_file_name = get_path_to_graph(graph_name)
    if not file_exists(output_graph_file_name):
        download_graph(graph_name)

        tar_name = SOURCE_GRAPH_DIR + snap_links[graph_name]
        cmd = ["gunzip", "-f", tar_name]
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        source_name = os.path.splitext(tar_name)[0]

        cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-directed",
               "-file", output_graph_file_name]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        verify_graph_existence(output_graph_file_name)
    else:
        print("Graph " + output_graph_file_name + " already exists")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-undirected",
                   "-file", output_graph_file_name]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            verify_graph_existence(output_graph_file_name)
        else:
            print("Graph " + output_graph_file_name + " already exists")


def create_synthetic_graph(graph_name, arch):
    dat = graph_name.split("_")
    type = dat[0]
    scale = dat[1]
    edge_factor = dat[2]

    output_graph_file_name = get_path_to_graph(graph_name)
    if not file_exists(output_graph_file_name):
        cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor, "-type", type, "-directed",
               "-file", output_graph_file_name]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        verify_graph_existence(output_graph_file_name)
    else:
        print("Graph " + output_graph_file_name + " already exists")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor,
                   "-type", type, "-undirected", "-file", output_graph_file_name]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            verify_graph_existence(output_graph_file_name)
        else:
            print("Graph " + output_graph_file_name + " already exists")


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
        create_graph(current_graph, arch)

