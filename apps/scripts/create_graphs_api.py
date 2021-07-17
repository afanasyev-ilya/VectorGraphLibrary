from .helpers import *
from .common import *
import os.path
from os import path


# real-world graphs
def get_list_of_social_graphs():
    graphs = ["soc_pokec"]
    return graphs


def get_list_of_web_graphs():
    graphs = ["web_baidu"]
    return graphs


def get_list_of_road_graphs():
    graphs = ["road_california"]
    return graphs


# synthetic graphs
def get_list_of_rmat_graphs():
    graphs = ["syn_rmat_20_32"]
    return graphs


def get_list_of_uniform_random_graphs():
    graphs = ["syn_ru_20_32"]
    return graphs


# all graphs
def get_list_of_synthetic_graphs():
    return get_list_of_rmat_graphs() + get_list_of_uniform_random_graphs()


def get_list_of_real_world_graphs():
    return get_list_of_social_graphs() + get_list_of_web_graphs() + get_list_of_road_graphs()


def get_list_of_all_graphs():
    return get_list_of_synthetic_graphs() + get_list_of_real_world_graphs()


def get_list_of_verification_graphs():
    synthetic_graph = get_list_of_synthetic_graphs()[0]
    social_graph = get_list_of_social_graphs()[0]
    web_graph = get_list_of_web_graphs()[0]
    road_graph = get_list_of_road_graphs()[0]
    return [synthetic_graph, social_graph, web_graph, road_graph]


def download_all_real_world_graphs():
    real_world_graphs = get_list_of_real_world_graphs()
    for graph_name in real_world_graphs:
        download_graph(graph_name)


def download_graph(graph_name):
    if not path.exists(SOURCE_GRAPH_DIR + "/" + konect_links[graph_name]):
        print("Trying to download " + SOURCE_GRAPH_DIR + konect_links[graph_name] + " using " +
              "http://konect.cc/files/download.tsv." + konect_links[graph_name] + ".tar.bz2")

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
        link = "http://konect.cc/files/download.tsv." + konect_links[graph_name] + ".tar.bz2"
        cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

    file_name = SOURCE_GRAPH_DIR + "download.tsv." + konect_links[graph_name] + ".tar.bz2"
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

        tar_name = SOURCE_GRAPH_DIR + "download.tsv." + konect_links[graph_name] + ".tar.bz2"
        cmd = ["tar", "-xjf", tar_name, '-C', "./bin/source_graphs"]
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        source_name = "./bin/source_graphs/" + konect_links[graph_name] + "/out." + konect_links[graph_name]
        #os.path.splitext(tar_name)[0]

        cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-directed",
               "-file", output_graph_file_name]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        verify_graph_existence(output_graph_file_name)

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-undirected",
                   "-file", output_graph_file_name]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            verify_graph_existence(output_graph_file_name)


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
        print("Graph " + output_graph_file_name + " has been created")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor,
                   "-type", type, "-undirected", "-file", output_graph_file_name]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            verify_graph_existence(output_graph_file_name)
            print("Graph " + output_graph_file_name + " has been created")


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

