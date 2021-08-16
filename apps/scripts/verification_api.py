from .helpers import *
from .create_graphs_api import *
from .common import *
from .export_results import *


def check_app_correctness(output):
    matched_lines = []
    for line in output.split("\n"):
        if correctness_pattern in line:
            matched_lines += [line]
    if not matched_lines:
        print("Warning! no correctness data found in current test.")
    return matched_lines


def verify_app(app_name, arch, benchmarking_results, graph_format):
    list_of_graphs = get_list_of_verification_graphs()

    create_graphs_if_required(list_of_graphs, arch, graph_format)
    common_args = ["-check", "-it", "1", "-format", graph_format]

    for current_args in benchmark_args[app_name]:
        benchmarking_results.add_correctness_test_name_to_xls_table(app_name, current_args + common_args)

        for current_graph in list_of_graphs:
            cmd = [get_binary_path(app_name, arch), "-load",
                   get_path_to_graph(current_graph, graph_format, requires_undir_graphs(app_name))] + current_args + common_args
            print(' '.join(cmd))
            proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
            output = proc.stdout.read().decode("utf-8")

            correctness_lines = check_app_correctness(output)
            print(correctness_lines)

            benchmarking_results.add_correctness_value_to_xls_table(str(correctness_lines), current_graph, app_name)

        benchmarking_results.add_performance_separator_to_xls_table()