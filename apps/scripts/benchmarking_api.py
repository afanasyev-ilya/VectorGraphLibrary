from .helpers import *
from .create_graphs_api import *
from .xls_stats_api import *
from .common import *
import re


def find_perf_line(output):
    matched_lines = []
    for line in output.split("\n"):
        #print(line)
        if perf_pattern in line:
            matched_lines += [line]
    if not matched_lines:
        print("Warning! no performance data found in current test.")
    return matched_lines


def extract_perf_val(perf_lines):
    if len(perf_lines) > 1:
        print("Warning! multiple performance data found in current test.")
    for line in perf_lines:
        value = re.findall("\d+\.\d+", line)
        if len(value) > 0:
            return float(value[0])
        else:
            print("Warning! perf line does not contain float in current test.")
    return 0.0


def benchmark_app(app_name, arch, options, workbook, table_stats):
    list_of_graphs = get_list_of_rmat_graphs() + get_list_of_ru_graphs() + get_list_of_soc_graphs() + \
                     get_list_of_misc_graphs()

    create_graphs_if_required(list_of_graphs, arch)
    common_args = ["-it", str(common_iterations)]

    for current_args in benchmark_args[app_name]:
        table_stats.init_test_data(app_name, current_args + common_args)

        for current_graph in list_of_graphs:
            cmd = [get_binary_path(app_name, arch), "-load", get_path_to_graph(current_graph)] + current_args + common_args
           print(' '.join(cmd))
            proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
            output = proc.stdout.read().decode("utf-8")

            perf_value = extract_perf_val(find_perf_line(output))
            table_stats.add_perf_value(perf_value, current_graph)

        table_stats.end_test_data()