import os
import optparse
from scripts.helpers import *
from scripts.benchmarking_api import *
from scripts.verification_api import *


def run_tests(options):
    create_dir("./bin/")
    list_of_apps = options.apps.split(",")
    arch = options.arch

    workbook = xlsxwriter.Workbook('benchmarking_results.xlsx')

    set_omp_environments(options)

    for app_name in list_of_apps:
        if is_valid(app_name, arch, options):
            benchmark_app(app_name, arch, options, workbook)
            if options.verify:
                verify_app(app_name, arch, options, workbook)

    workbook.close()


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
    parser.add_option('-c', '--compile',
                      action="store_true", dest="recompile",
                      help="recompile all binaries used in testing", default=False)
    parser.add_option('-v', '--verify',
                      action="store_true", dest="verify",
                      help="run additional verification tests after benchmarking process", default=False)

    options, args = parser.parse_args()
    run_tests(options)
