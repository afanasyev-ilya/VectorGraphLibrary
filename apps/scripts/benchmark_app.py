from .helpers import *
from .create_graphs_api import *
from .xls_stats_api import *


def find_perf_line(output):
    matched_lines = []
    for line in output.split("\n"):
        print(line)
        if "MAX_PERF" in line:
            matched_lines += [line]
    return matched_lines


def benchmark_app(app_name, arch, options):
    list_of_graphs = get_list_of_soc_graphs() + get_list_of_misc_graphs() + get_list_of_rmat_graphs() + \
                         get_list_of_ru_graphs()

    create_graphs_if_required(list_of_graphs, arch)

    for current_graph in list_of_graphs:
        cmd = [get_binary_path(app_name, arch), "--load", get_graph_path(current_graph)]
        print(cmd)
        proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
        output = proc.stdout.read().decode("utf-8")
        save_perf_stats(find_perf_line(output))