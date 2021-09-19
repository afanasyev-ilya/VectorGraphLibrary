from .helpers import *
from .settings import *
import os.path
from os import path


# synthetic
def get_list_of_synthetic_graphs(run_speed_mode):
    if run_speed_mode == "tiny-only":
        return syn_tiny_only
    elif run_speed_mode == "small-only":
        return syn_small_only
    elif run_speed_mode == "medium-only":
        return syn_medium_only
    elif run_speed_mode == "large-only":
        return syn_large_only
    elif run_speed_mode == "rating-fast":
        return syn_rating_fast
    elif run_speed_mode == "rating-medium":
        return syn_rating_medium
    elif run_speed_mode == "rating-full":
        return syn_rating_full
    elif run_speed_mode == "tiny-small":
        return syn_tiny_small
    elif run_speed_mode == "tiny-small-medium":
        return syn_tiny_small_medium
    else:
        raise ValueError("Unsupported run_speed_mode")


def get_list_of_real_world_graphs(run_speed_mode):
    if run_speed_mode == "tiny-only":
        return konect_tiny_only
    elif run_speed_mode == "small-only":
        return konect_small_only
    elif run_speed_mode == "medium-only":
        return konect_medium_only
    elif run_speed_mode == "large-only":
        return konect_large_only
    elif run_speed_mode == "rating-fast":
        return konect_rating_fast
    elif run_speed_mode == "rating-medium":
        return konect_rating_medium
    elif run_speed_mode == "rating-full":
        return konect_rating_full
    elif run_speed_mode == "tiny-small":
        return konect_tiny_small
    elif run_speed_mode == "tiny-small-medium":
        return konect_tiny_small_medium
    else:
        raise ValueError("Unsupported run_speed_mode")


def get_list_of_all_graphs(run_speed_mode):
    return get_list_of_synthetic_graphs(run_speed_mode) + get_list_of_real_world_graphs(run_speed_mode)


def get_list_of_verification_graphs(run_speed_mode):
    verification_list = []
    i = 0
    for graph in get_list_of_synthetic_graphs(run_speed_mode):
        if i >= 3:
            break
        verification_list.append(graph)
        i += 1
    i = 0
    for graph in get_list_of_real_world_graphs(run_speed_mode):
        if i >= 5:
            break
        verification_list.append(graph)
        i += 1
    return verification_list


def download_all_real_world_graphs(run_speed_mode):
    real_world_graphs = get_list_of_real_world_graphs(run_speed_mode)
    for graph_name in real_world_graphs:
        download_graph(graph_name)


def download_graph(graph_name):
    file_name = SOURCE_GRAPH_DIR + "download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
    if not path.exists(file_name):
        link = "http://konect.cc/files/download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
        print("Trying to download " + file_name + " using " + link)
        cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)
        if path.exists(file_name):
            print(file_name + " downloaded!")
        else:
            print("Error! Can not download file " + file_name)
    else:
        print("File " + SOURCE_GRAPH_DIR + "/" + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2" + " exists!")


def get_path_to_graph(short_name, graph_format, undir = False):
    if undir:
        return GRAPHS_DIR + UNDIRECTED_PREFIX + short_name + ".vgraph." + graph_format
    else:
        return GRAPHS_DIR + short_name + ".vgraph." + graph_format


def verify_graph_existence(graph_file_name):
    if file_exists(graph_file_name):
        print("Success! graph " + graph_file_name + " has been created.")
        return True
    else:
        print("Error! graph " + graph_file_name + " can not be created.")
        return False


def create_real_world_graph(graph_name, arch):
    graph_format = "el_container"
    output_graph_file_name = get_path_to_graph(graph_name, graph_format)
    if not file_exists(output_graph_file_name):
        download_graph(graph_name)

        tar_name = SOURCE_GRAPH_DIR + "download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
        cmd = ["tar", "-xjf", tar_name, '-C', "./bin/source_graphs"]
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        if "unarch_graph_name" in all_konect_graphs_data[graph_name]:
            source_name = "./bin/source_graphs/" + all_konect_graphs_data[graph_name]["link"] + "/out." + all_konect_graphs_data[graph_name]["unarch_graph_name"]
        else:
            source_name = "./bin/source_graphs/" + all_konect_graphs_data[graph_name]["link"] + "/out." + all_konect_graphs_data[graph_name]["link"]

        cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-directed",
               "-file", output_graph_file_name.replace("."+graph_format, ""), "-format", graph_format]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        if verify_graph_existence(output_graph_file_name):
            print("Graph " + output_graph_file_name + " has been created\n")
    else:
        print("Warning! Graph " + output_graph_file_name + " already exists!")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, graph_format, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-convert", source_name, "-undirected",
                   "-file", output_graph_file_name.replace("."+graph_format, ""), "-format", graph_format]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            if verify_graph_existence(output_graph_file_name):
                print("Graph " + output_graph_file_name + " has been created\n")
        else:
            print("Warning! Graph " + output_graph_file_name + " already exists!")


def create_synthetic_graph(graph_name, arch):
    graph_format = "el_container"
    dat = graph_name.split("_")
    type = dat[1]
    scale = dat[2]
    edge_factor = dat[3]

    output_graph_file_name = get_path_to_graph(graph_name, graph_format)
    if not file_exists(output_graph_file_name):
        cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor, "-type", type, "-directed",
               "-file", output_graph_file_name.replace("."+graph_format, ""), "-format", graph_format]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
        if verify_graph_existence(output_graph_file_name):
            print("Graph " + output_graph_file_name + " has been created\n")
    else:
        print("Warning! Graph " + output_graph_file_name + " already exists!")

    if GENERATE_UNDIRECTED_GRAPHS:
        output_graph_file_name = get_path_to_graph(graph_name, graph_format, True)
        if not file_exists(output_graph_file_name):
            cmd = [get_binary_path("create_vgl_graphs", arch), "-s", scale, "-e", edge_factor,
                   "-type", type, "-undirected", "-file", output_graph_file_name.replace("."+graph_format, ""),
                   "-format", graph_format]
            print(' '.join(cmd))
            subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()
            if verify_graph_existence(output_graph_file_name):
                print("Graph " + output_graph_file_name + " has been created\n")
        else:
            print("Warning! Graph " + output_graph_file_name + " already exists!")


def create_graph(graph_name, arch, run_speed_mode):
    if graph_name in get_list_of_synthetic_graphs(run_speed_mode):
        create_synthetic_graph(graph_name, arch)
    elif graph_name in get_list_of_real_world_graphs(run_speed_mode):
        create_real_world_graph(graph_name, arch)


def create_graphs_if_required(list_of_graphs, arch, run_speed_mode):
    create_dir(GRAPHS_DIR)
    create_dir(SOURCE_GRAPH_DIR)

    if not binary_exists("create_vgl_graphs", arch):
        make_binary("create_vgl_graphs", arch)

    for current_graph in list_of_graphs:
        create_graph(current_graph, arch, run_speed_mode)

