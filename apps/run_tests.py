import os
import optparse
from scripts.helpers import *
from scripts.benchmark_app import *


def run_tests(options):
    create_dir("./bin/")
    list_of_apps = options.apps.split(",")
    arch = options.arch
    for app_name in list_of_apps:
        if is_valid(app_name, arch):
            benchmark_app(app_name, arch, options)


if __name__ == "__main__":
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-a', '--apps',
                      action="store", dest="apps",
                      help="specify an application to test (or all for testing all available applications)", default="all")
    parser.add_option('-r', '--arch',
                      action="store", dest="arch",
                      help="specify evaluated architecture: sx/aurora, mc/multicore, cu/gpu", default="sx")
    parser.add_option('-f', '--force',
                      action="store", dest="force",
                      help="specify parameters for running a specific application", default="")
    parser.add_option('-s', '--sockets',
                      action="store", dest="sockets",
                      help="set number of sockets used", default=1)
    parser.add_option('-c', '--csv',
                      action="store_true", dest="csv_results",
                      help="output performance results to CSV file", default=False)
    parser.add_option('-o', '--stdout',
                      action="store_true", dest="stdout_results",
                      help="use stdout to output performance results", default=False)

    options, args = parser.parse_args()

    run_tests(options)