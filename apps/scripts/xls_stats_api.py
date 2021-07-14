import xlsxwriter
from .create_graphs_api import *
from random import randrange
from .submit_results import submit


app_name_column_size = 30
graph_name_column_size = 20
data_column_size = 15
colors = ["#CCFFFF", "#CCFFCC", "#FFFF99", "#FF99FF", "#66CCFF", "#FF9966"]


def lines_in_test():
    sizes = [len(get_list_of_rmat_graphs()),
             len(get_list_of_ru_graphs()),
             len(get_list_of_soc_graphs()),
             len(get_list_of_misc_graphs())]
    return int(max(sizes))


def get_column_pos(graph_name):
    if "rmat" in graph_name:
        return 2
    elif "ru" in graph_name:
        return 4
    elif graph_name in get_list_of_soc_graphs():
        return 6
    elif graph_name in get_list_of_misc_graphs():
        return 8
    else:
        raise ValueError("Incorrect graph name")


def get_row_pos(graph_name):
    if "rmat" in graph_name or "ru" in graph_name:
        return int(graph_name.split('_')[1]) - synthetic_min_scale
    elif graph_name in get_list_of_soc_graphs():
        return get_list_of_soc_graphs().index(graph_name)
    elif graph_name in get_list_of_misc_graphs():
        return get_list_of_misc_graphs().index(graph_name)


class PerformanceStats:
    def __init__(self, workbook):
        self.workbook = workbook
        self.worksheet = self.workbook.add_worksheet("Performance data")
        self.line_pos = 0
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""
        self.perf_data = []

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

    def init_test_data(self, app_name, app_args):
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

        rmat_graphs = get_list_of_rmat_graphs()
        i = 0
        for graph_name in rmat_graphs:
            self.worksheet.write(self.line_pos + i, 1, graph_name, self.current_format)
            i += 1

        ru_graphs = get_list_of_ru_graphs()
        i = 0
        for graph_name in ru_graphs:
            self.worksheet.write(self.line_pos + i, 3, graph_name, self.current_format)
            i += 1

        soc_graphs = get_list_of_soc_graphs()
        i = 0
        for graph_name in soc_graphs:
            self.worksheet.write(self.line_pos + i, 5, graph_name, self.current_format)
            i += 1

        misc_graphs = get_list_of_misc_graphs()
        i = 0
        for graph_name in misc_graphs:
            self.worksheet.write(self.line_pos + i, 7, graph_name, self.current_format)
            i += 1

    def add_perf_value(self, perf_value, graph_name):
        row = int(get_row_pos(graph_name))
        col = int(get_column_pos(graph_name))
        #print(self.current_app_name + ", " + graph_name + ", " + str(perf_value))
        self.perf_data.append({"app": self.current_app_name, "graph": graph_name, "perf": perf_value})
        #print(perf_value)
        self.worksheet.write(self.line_pos + row, col, perf_value, self.current_format)

    def end_test_data(self):
        self.line_pos += lines_in_test() + 1

    def export_perf_data(self):
        submit("kunpeng 920 ilya", self.perf_data, ["test"])


class VerificationStats:
    def __init__(self, workbook):
        self.workbook = workbook
        self.worksheet = self.workbook.add_worksheet("Verification data")

        # add column names
        for graph_name in get_list_of_verification_graphs():
            self.worksheet.write(0, get_list_of_verification_graphs().index(graph_name) + 1, graph_name)

        self.worksheet.set_column(0, len(get_list_of_verification_graphs()) + 1, 30)

        self.current_pos = 1

    def init_test_data(self, app_name, app_args):
        test_name = ' '.join([app_name] + app_args)
        self.worksheet.write(self.current_pos, 0, test_name)

    def end_test_data(self):
        self.current_pos += 1

    def add_correctness_value(self, data, graph_name):
        self.worksheet.write(self.current_pos, get_list_of_verification_graphs().index(graph_name) + 1, data)
