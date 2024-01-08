from .helpers import *
from .create_graphs_api import *
from .export import *
from .settings import *
import re
import time
from threading import Timer
import pickle
import numpy as np
import sklearn
#from sklearn.ensemble import RandomForestClassifier


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


def benchmark_app(app_name, arch, benchmarking_results, graph_format, run_speed_mode, timeout_length):
    list_of_graphs = get_list_of_all_graphs(run_speed_mode)

    create_graphs_if_required(list_of_graphs, arch, run_speed_mode)
    print(graph_format)
    loaded_model_filename = str(arch) + "_" + str(app_name) + "_model.sav"
    features_filename =  str(arch) + "_" + str(app_name) + "_features.pickle"
    loaded_model = None
    features_map = None
    if graph_format == "ml":
        common_args = ["-it", str(common_iterations)]
        loaded_model = pickle.load(open(loaded_model_filename, 'rb'))
        features_map = pickle.load(open(features_filename, 'rb'))
    else:
        common_args = ["-it", str(common_iterations), "-format", graph_format]

    algorithms_tested = 0

    for current_args in benchmark_args[app_name]:
        benchmarking_results.add_performance_test_name_to_xls_table(app_name, current_args + common_args)

        for current_graph in list_of_graphs:
            start = time.time()
            if app_name in apps_and_graphs_ingore and current_graph in apps_and_graphs_ingore[app_name]:
                print("graph " + current_graph + " is set to be ignored for app " + app_name + "!")
                continue
            
            if graph_format == "ml":
                features = features_map[current_graph]
                pred = loaded_model.predict(np.array(features).reshape(-1,5))[0]
                common_args.append("-format")
                common_args.append(gpu_bfs_format_label[pred])
                
            cmd = [get_binary_path(app_name, arch), "-import", get_path_to_graph(current_graph, "el_container", requires_undir_graphs(app_name))] + current_args + common_args
            print(' '.join(cmd))
            proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            timer = Timer(int(timeout_length), proc.kill)
            try:
                timer.start()
                stdout, stderr = proc.communicate()
            finally:
                timer.cancel()

            if graph_format == "ml":
                common_args = ["-it", str(common_iterations)]

            output = stdout.decode("utf-8")

            perf_value = extract_perf_val(find_perf_line(output))

            if perf_value == 0.0:
                perf_value = "TIMED OUT"

            benchmarking_results.add_performance_value_to_xls_table(perf_value, current_graph, app_name)
            end = time.time()
            if print_timings:
                print("TIME: " + str(end-start) + " seconds\n")

        benchmarking_results.add_performance_separator_to_xls_table()
        algorithms_tested += 1

    return algorithms_tested
