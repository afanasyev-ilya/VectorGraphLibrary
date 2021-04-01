import os
import optparse
from scripts.helpers import *
from scripts.benchmarking_api import *
from scripts.verification_api import *


def run_compile(options, arch):
    make_binary("clean", arch)
    list_of_apps = prepare_list_of_apps(options.apps)
    for app_name in list_of_apps:
        make_binary(app_name, arch)
        if not binary_exists(app_name, arch):
            print("Error! Can not compile " + app_name + ", several errors occurred.")


def run_prepare(options, arch):
    create_graphs_if_required(get_list_of_all_graphs(), arch)


def run_tests(options, arch):
    list_of_apps = prepare_list_of_apps(options.apps)

    stats_file_name = "benchmarking_results_" + arch + ".xlsx"
    workbook = xlsxwriter.Workbook(stats_file_name)

    set_omp_environments(options)
    perf_stats = PerformanceStats(workbook)
    verification_stats = VerificationStats(workbook)

    for app_name in list_of_apps:
        if is_valid(app_name, arch, options):
            if not options.no_run:
                benchmark_app(app_name, arch, options, workbook, perf_stats)
            if options.verify:
                verify_app(app_name, arch, options, workbook, verification_stats)
        else:
            print("Error! Can not compile " + app_name + ", several errors occurred.")

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
    parser.add_option('-v', '--verify',
                      action="store_true", dest="verify",
                      help="run additional verification tests after benchmarking process", default=False)
    parser.add_option('-c', '--compile',
                      action="store_true", dest="compile",
                      help="compile all binaries", default=False)
    parser.add_option('-p', '--prepare',
                      action="store_true", dest="prepare",
                      help="compile all binaries, download graphs and convert them into VGL format", default=False)
    parser.add_option('-n', '--no-run',
                      action="store_true", dest="no_run",
                      help="do not run any tests", default=False)

    options, args = parser.parse_args()

    create_dir("./bin/")
    arch = options.arch

    if options.compile:
        run_compile(options, arch)

    if options.prepare:
        run_prepare(options, arch)

    run_tests(options, arch)
