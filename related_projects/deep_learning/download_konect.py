import os
import optparse
import subprocess
import pickle
import sys
from os import path


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def condition(graph):
    if pow(2, 14) < graph["size"] < pow(2, 21):
        return True
    return False


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def remove_suffix(text, prefix):
    return text[text.endswith(prefix) and len(prefix):]


def process_graph(graph_name, graph_data):
    if "http" in graph_data["tsv_link"]:
        print(graph_data)
        suffix = graph_data["tsv_link"].split('/')[-1]
        print(suffix)

        file_name = "./obj/" + suffix
        if not path.exists(file_name):
            link = graph_data["tsv_link"]
            print("Trying to download " + link)
            cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", "./obj"]
            print(' '.join(cmd))
            subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)
            if path.exists(file_name):
                print(file_name + " downloaded!")
            else:
                print("Error! Can not download file " + file_name)
        else:
            print("File " + "./obj/" + suffix + " exists!")

        tar_name = "./obj/" + suffix
        cmd = ["tar", "-xjf", tar_name, '-C', "./obj/"]
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        sub_path = suffix.replace('.tar.bz2', '')
        sub_path = sub_path.replace('download.tsv.', '')

        print(sub_path)
        source_name = "./obj/" + sub_path + "/out." + sub_path
        cmd = ["./../../apps/bin/create_vgl_graphs_mc", "-convert", source_name, "-directed",
               "-file", "./source_graphs/" + graph_name.replace(" ", "_"), "-format", "el_container"]
        print(' '.join(cmd))
        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()


def download_learning_data():
    graphs = {}
    unsorted_dict = load_obj("graphs")
    for key in unsorted_dict.keys():
        graph_name = key
        graph_data = unsorted_dict[key]
        if condition(graph_data):
            print(graph_name + " satisfies condition")
            process_graph(graph_name, graph_data)


if __name__ == "__main__":
    download_learning_data()




