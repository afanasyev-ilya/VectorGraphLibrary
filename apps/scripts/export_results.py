import xlsxwriter
from .create_graphs_api import *
from random import randrange
from .submit_results import submit_to_socket
import pickle


app_name_column_size = 30
graph_name_column_size = 20
data_column_size = 15
colors = ["#CCFFFF", "#CCFFCC", "#FFFF99", "#FF99FF", "#66CCFF", "#FF9966"]


def lines_in_test():
    return int(max(len(get_list_of_synthetic_graphs()), len(get_list_of_real_world_graphs())))


def get_column_pos(graph_name):
    if graph_name in get_list_of_synthetic_graphs():
        return 2
    elif graph_name in get_list_of_real_world_graphs():
        return 4
    else:
        raise ValueError("Incorrect graph name")


def get_row_pos(graph_name):
    if graph_name in get_list_of_synthetic_graphs():
        return get_list_of_synthetic_graphs().index(graph_name)
    elif graph_name in get_list_of_real_world_graphs():
        return get_list_of_real_world_graphs().index(graph_name)


class BenchmarkingResults:
    def __init__(self):
        self.performance_data = []
        self.correctness_data = []

        self.workbook = xlsxwriter.Workbook("benchmarking_results.xlsx")
        self.worksheet = None # these can be later used for xls output
        self.line_pos = None # these can be later used for xls output
        self.current_format = None # these can be later used for xls output
        self.current_app_name = None # these can be later used for xls output

    def add_performance_header_to_xls_table(self):
        self.worksheet = self.workbook.add_worksheet("Performance data")
        self.line_pos = 0
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""

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

        self.worksheet.merge_range(self.line_pos, 0, self.line_pos + lines_in_test() - 1, 0,
                                   test_name, self.current_format)

    def add_performance_value_to_xls_table(self, perf_value, graph_name, app_name):
        row = int(get_row_pos(graph_name))
        col = int(get_column_pos(graph_name))

        self.worksheet.write(self.line_pos + row, col - 1, graph_name, self.current_format)
        self.worksheet.write(self.line_pos + row, col, perf_value, self.current_format)
        self.performance_data.append({"graph_name": graph_name, "app_name": app_name, "perf_val": perf_value})

    def add_performance_separator_to_xls_table(self):
        self.line_pos += lines_in_test() + 1

    def add_correctness_header_to_xls_table(self):
        self.worksheet = self.workbook.add_worksheet("Correctness data")
        self.line_pos = 0
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""

        # add column names
        for graph_name in get_list_of_verification_graphs():
            self.worksheet.write(self.line_pos, get_list_of_verification_graphs().index(graph_name) + 1, graph_name)

        self.worksheet.set_column(self.line_pos, len(get_list_of_verification_graphs()) + 1, 30)

        self.line_pos = 1

    def add_correctness_test_name_to_xls_table(self, app_name, app_args):
        test_name = ' '.join([app_name] + app_args)
        self.worksheet.write(self.line_pos, 0, test_name)

    def add_correctness_separator_to_xls_table(self):
        self.line_pos += 1

    def add_correctness_value_to_xls_table(self, value, graph_name, app_name):
        self.worksheet.write(self.line_pos, get_list_of_verification_graphs().index(graph_name) + 1, value)
        self.correctness_data.append({"graph_name": graph_name, "app_name": app_name, "correctness_val": value})

    def submit(self, arch):
        send_dict = {"arch_name": arch, "performance_data": self.performance_data, "correctness_data": self.correctness_data}
        submit_to_socket(send_dict)

    def offline_submit(self, arch, run_name):
        send_dict = {"arch_name": arch, "performance_data": self.performance_data, "correctness_data": self.correctness_data}
        # send_dict
        a_file = open(run_name + "_vgl_rating_data.pkl", "wb")
        pickle.dump(send_dict, a_file)
        a_file.close()

    def finalize(self):
        self.workbook.close()
