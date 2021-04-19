from .helpers import *
from .create_graphs_api import *
from .common import *
from .xls_stats_api import *


def check_app_correctness(output):
    matched_lines = []
    for line in output.split("\n"):
        if correctness_pattern in line:
            matched_lines += [line]
    if not matched_lines:
        print("Warning! no correctness data found in current test.")
    return matched_lines


def verify_app(app_name, arch, options, workbook, table_stats):
    list_of_graphs = get_list_of_verification_graphs()

    create_graphs_if_required(list_of_graphs, arch)
    common_args = ["-check", "-it", "1"]

    for current_args in benchmark_args[app_name]:
        table_stats.init_test_data(app_name, current_args)

        for current_graph in list_of_graphs:
            cmd = [get_binary_path(app_name, arch), "-load",
                   get_path_to_graph(current_graph, requires_undir_graphs(app_name))] + current_args + common_args
           print(' '.join(cmd))
            proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
            output = proc.stdout.read().decode("utf-8")

            correctness_lines = check_app_correctness(output)
            print(correctness_lines)

            table_stats.add_correctness_value(str(correctness_lines), current_graph)

        table_stats.end_test_data()