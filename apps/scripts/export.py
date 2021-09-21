import xlsxwriter
from .create_graphs_api import *
from random import randrange
from .submit_results import submit_to_socket
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil


app_name_column_size = 30
graph_name_column_size = 20
data_column_size = 15
colors = ["#CCFFFF", "#CCFFCC", "#FFFF99", "#FF99FF", "#66CCFF", "#FF9966"]


def remove_timed_out(perf_data):
    cleared_list = []
    for item in perf_data:
        if item["perf_val"] != "TIMED OUT":
            cleared_list.append(item)
    return cleared_list


class BenchmarkingResults:
    def __init__(self, name, run_speed_mode):
        self.performance_data = []
        self.correctness_data = []
        self.run_speed_mode = run_speed_mode
        self.current_graph_format = ""

        self.workbook = xlsxwriter.Workbook(name + "_benchmarking_results.xlsx")
        self.worksheet = None # these can be later used for xls output
        self.line_pos = None # these can be later used for xls output
        self.current_format = None # these can be later used for xls output
        self.current_app_name = None # these can be later used for xls output

    def add_performance_header_to_xls_table(self, graph_format):
        self.worksheet = self.workbook.add_worksheet("Perf " + graph_format)
        self.line_pos = 0
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""
        self.current_graph_format = graph_format

        # make columns wider
        self.worksheet.set_column(0, 0, app_name_column_size)
        self.worksheet.set_column(1, 1, graph_name_column_size)
        self.worksheet.set_column(2, 2, data_column_size)
        self.worksheet.set_column(3, 3, graph_name_column_size)
        self.worksheet.set_column(4, 5, data_column_size)
        self.worksheet.set_column(5, 5, graph_name_column_size)
        self.worksheet.set_column(6, 6, data_column_size)
        self.worksheet.set_column(7, 7, graph_name_column_size)
        self.worksheet.set_column(8, 8, data_column_size)

    def add_performance_test_name_to_xls_table(self, app_name, app_args):
        test_name = ' '.join([app_name] + app_args)
        self.worksheet.write(self.line_pos, 0, test_name)
        self.current_app_name = app_name

        color = colors[randrange(len(colors))]
        self.current_format = self.workbook.add_format({'border': 1,
                                                        'align': 'center',
                                                        'valign': 'vcenter',
                                                        'fg_color': color})

        self.worksheet.merge_range(self.line_pos, 0, self.line_pos + self.lines_in_test() - 1, 0,
                                   test_name, self.current_format)

    def add_performance_value_to_xls_table(self, perf_value, graph_name, app_name):
        row = int(self.get_row_pos(graph_name))
        col = int(self.get_column_pos(graph_name))

        self.worksheet.write(self.line_pos + row, col - 1, graph_name, self.current_format)
        self.worksheet.write(self.line_pos + row, col, perf_value, self.current_format)
        self.performance_data.append({"graph_name": graph_name, "app_name": app_name, "perf_val": perf_value,
                                      "format": self.current_graph_format})

    def add_performance_separator_to_xls_table(self):
        self.line_pos += self.lines_in_test() + 1

    def add_correctness_header_to_xls_table(self, graph_format):
        self.worksheet = self.workbook.add_worksheet("Correctness data " + graph_format)
        self.line_pos = 0
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""

        # add column names
        for graph_name in get_list_of_verification_graphs(self.run_speed_mode):
            self.worksheet.write(self.line_pos, get_list_of_verification_graphs(self.run_speed_mode).index(graph_name) + 1, graph_name)

        self.worksheet.set_column(self.line_pos, len(get_list_of_verification_graphs(self.run_speed_mode)) + 1, 30)

        self.line_pos = 1

    def add_correctness_test_name_to_xls_table(self, app_name, app_args):
        test_name = ' '.join([app_name] + app_args)
        self.worksheet.write(self.line_pos, 0, test_name)

    def add_correctness_separator_to_xls_table(self):
        self.line_pos += 1

    def add_correctness_value_to_xls_table(self, value, graph_name, app_name):
        self.worksheet.write(self.line_pos, get_list_of_verification_graphs(self.run_speed_mode).index(graph_name) + 1, value)
        self.correctness_data.append({"graph_name": graph_name, "app_name": app_name, "correctness_val": value,
                                      "format": self.current_graph_format})

    def plot(self, formats_list):
        tested_apps = []
        row_data = remove_timed_out(self.performance_data)

        for item in row_data:
            if item["app_name"] not in tested_apps:
                tested_apps.append(item["app_name"])

        if os.path.exists("./plots"):
            shutil.rmtree("./plots")

        os.makedirs("./plots")

        for app_name in tested_apps:
            for format_name in formats_list:
                plot_names = []
                perf_vals = []
                x_vals = []
                it = 0
                for item in row_data:
                    if item["app_name"] == app_name and item["format"] == self.current_graph_format:
                        plot_names.append(item["graph_name"])
                        perf_vals.append(item["perf_val"])
                        x_vals.append(it)
                        it += 1

                x = np.array(x_vals)
                y = np.array(perf_vals)
                xticks = plot_names

                plt.ylabel('Performance (MTEPS)')
                plt.xticks(x, xticks, rotation=90)
                plt.plot(x, y, label=format_name)
            plt.savefig("./plots/" + app_name + ".png", bbox_inches='tight')

    def submit(self, run_info):
        send_dict = {"run_info": run_info, "performance_data": remove_timed_out(self.performance_data), "correctness_data": self.correctness_data}
        return submit_to_socket(send_dict)

    def offline_submit(self, run_info, name):
        send_dict = {"run_info": run_info, "performance_data": remove_timed_out(self.performance_data), "correctness_data": self.correctness_data}
        # send_dict
        a_file = open(name + "_vgl_rating_data.pkl", "wb")
        pickle.dump(send_dict, a_file)
        a_file.close()

    def finalize(self):
        self.workbook.close()

    def lines_in_test(self):
        return int(max(len(get_list_of_synthetic_graphs(self.run_speed_mode)), len(get_list_of_real_world_graphs(self.run_speed_mode))))

    def get_column_pos(self, graph_name):
        if graph_name in get_list_of_synthetic_graphs(self.run_speed_mode):
            return 2
        elif graph_name in get_list_of_real_world_graphs(self.run_speed_mode):
            return 4
        else:
            raise ValueError("Incorrect graph name")

    def get_row_pos(self, graph_name):
        if graph_name in get_list_of_synthetic_graphs(self.run_speed_mode):
            return get_list_of_synthetic_graphs(self.run_speed_mode).index(graph_name)
        elif graph_name in get_list_of_real_world_graphs(self.run_speed_mode):
            return get_list_of_real_world_graphs(self.run_speed_mode).index(graph_name)
