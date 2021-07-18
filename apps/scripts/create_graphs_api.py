from .helpers import *
from .common import *
import os.path
from os import path


# synthetic graphs
def get_list_of_rmat_graphs():
    graphs = ["syn_rmat_18_32", "syn_rmat_20_32", "syn_rmat_22_32", "syn_rmat_23_32", "syn_rmat_25_32"]
    return graphs


def get_list_of_uniform_random_graphs():
    graphs = ["syn_ru_18_32", "syn_ru_20_32", "syn_ru_22_32", "syn_ru_23_32"]
    return graphs


# all graphs
def get_list_of_synthetic_graphs():
    return get_list_of_rmat_graphs() + get_list_of_uniform_random_graphs()


def get_list_of_real_world_graphs():
    return list(konect_graphs_data.keys())


def get_list_of_all_graphs():
    return get_list_of_synthetic_graphs() + get_list_of_real_world_graphs()


def get_list_of_verification_graphs():
    synthetic_graph = "syn_rmat_20_32"
    social_graph = "soc_pokec"
    web_graph = "web_baidu"
    road_graph = "road_california"
    return [synthetic_graph, social_graph, web_graph, road_graph]


def download_all_real_world_graphs():
    real_world_graphs = get_list_of_real_world_graphs()
    for graph_name in real_world_graphs:
        download_graph(graph_name)


def download_graph(graph_name):
    if not path.exists(SOURCE_GRAPH_DIR + "/" + konect_graphs_data[graph_name]["link"]):
        print("Trying to download " + SOURCE_GRAPH_DIR + konect_graphs_data[graph_name]["link"] + " using " +
              "http://konect.cc/files/download.tsv." + konect_graphs_data[graph_name]["link"] + ".tar.bz2")

        #if not internet_on():
        #    print("Error! no internet connection available.")
        #    return

        # old snap usage
        #link = "https://snap.stanford.edu/data/" + snap_links[graph_name]
        #cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        #print(' '.join(cmd))
        #subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        #link = "https://snap.stanford.edu/data/bigdata/communities/" + snap_links[graph_name]
        #cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        #print(' '.join(cmd))
        #subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        # new konect usage
        link = "http://konect.cc/files/download.tsv." + konect_graphs_data[graph_name]["link"] + ".tar.bz2"
        cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

    file_name = SOURCE_GRAPH_DIR + "download.tsv." + konect_graphs_data[graph_name]["link"] + ".tar.bz2"
    if path.exists(file_name):
        print(file_name + " downloaded!")
    else:
        print("Error! Can not download file " + file_name)


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

        tar_name = SOURCE_GRAPH_DIR + "download.tsv." + konect_graphs_data[graph_name]["link"] + ".tar.bz2"
        cmd = ["tar", "-xjf", tar_name, '-C', "./bin/source_graphs"]
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        source_name = "./bin/source_graphs/" + konect_graphs_data[graph_name]["link"] + "/out." + konect_graphs_data[graph_name]["link"]
        #os.path.splitext(tar_name)[0]

        cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-directed",
               "-file", output_graph_file_name]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        verify_graph_existence(output_graph_file_name)
        print("Graph " + output_graph_file_name + " has been created\n")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-undirected",
                   "-file", output_graph_file_name]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            verify_graph_existence(output_graph_file_name)
            print("Graph " + output_graph_file_name + " has been created\n")


def create_synthetic_graph(graph_name, arch):
    dat = graph_name.split("_")
    type = dat[1]
    scale = dat[2]
    edge_factor = dat[3]

    output_graph_file_name = get_path_to_graph(graph_name)
    if not file_exists(output_graph_file_name):
        cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor, "-type", type, "-directed",
               "-file", output_graph_file_name]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        verify_graph_existence(output_graph_file_name)
        print("Graph " + output_graph_file_name + " has been created\n")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor,
                   "-type", type, "-undirected", "-file", output_graph_file_name]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            verify_graph_existence(output_graph_file_name)
            print("Graph " + output_graph_file_name + " has been created\n")


def create_graph(graph_name, arch):
    if graph_name in get_list_of_synthetic_graphs():
        create_synthetic_graph(graph_name, arch)
    elif graph_name in get_list_of_real_world_graphs():
        create_real_world_graph(graph_name, arch)


def create_graphs_if_required(list_of_graphs, arch):
    create_dir(GRAPHS_DIR)
    create_dir(SOURCE_GRAPH_DIR)

    if not binary_exists("create_vgl_graphs", arch):
        make_binary("create_vgl_graphs", arch)

    for current_graph in list_of_graphs:
        create_graph(current_graph, arch)

