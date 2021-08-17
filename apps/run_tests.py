import os
import optparse
from scripts.benchmarking_api import *
from scripts.verification_api import *
from scripts.export_results import BenchmarkingResults


def run_compile(options, arch):
    make_binary("clean", arch)
    list_of_apps = prepare_list_of_apps(options.apps)
    for app_name in list_of_apps:
        make_binary(app_name, arch)
        if not binary_exists(app_name, arch):
            print("Error! Can not compile " + app_name + ", several errors occurred.")


def run_prepare(options, arch):
    create_graphs_if_required(get_list_of_all_graphs(), arch, options.format)


def run_benchmarks(options, arch, benchmarking_results):
    list_of_apps = prepare_list_of_apps(options.apps)

    set_omp_environments(options)
    benchmarking_results.add_performance_header_to_xls_table()

    for app_name in list_of_apps:
        if is_valid(app_name, arch, options):
            benchmark_app(app_name, arch, benchmarking_results, options.format)
        else:
            print("Error! Can not benchmark " + app_name + ", several errors occurred.")


def run_verify(options, arch, benchmarking_results):
    list_of_apps = prepare_list_of_apps(options.apps)

    set_omp_environments(options)
    benchmarking_results.add_correctness_header_to_xls_table()

    for app_name in list_of_apps:
        if is_valid(app_name, arch, options):
            verify_app(app_name, arch, benchmarking_results, options.format)
        else:
            print("Error! Can not compile " + app_name + ", several errors occurred.")


def run(options):
    create_dir("./bin/")
    arch = options.arch

    benchmarking_results = BenchmarkingResults()

    if options.compile:
        run_compile(options, arch)

    if options.prepare:
        run_prepare(options, arch)

    if options.benchmark:
        run_benchmarks(options, arch, benchmarking_results)

    if options.verify:
        run_verify(options, arch, benchmarking_results)

    if options.submit is not None:
        if benchmarking_results.submit(options.submit):
            print("Results sent to server!")
        else:
            print("Can not send results, saving to file...")
            benchmarking_results.offline_submit(options.submit)

    benchmarking_results.finalize()


def main():
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-a', '--apps',
                      action="store", dest="apps",
                      help="specify an application to test (or all for testing all available applications)", default="all")
    parser.add_option('-r', '--arch',
                      action="store", dest="arch",
                      help="specify evaluated architecture: sx/aurora, mc/multicore, cu/gpu", default="sx")
    parser.add_option('-f', '--format',
                      action="store", dest="format",
                      help="specify graph storage format used", default="vcsr")
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
    parser.add_option('-b', '--benchmark',
                      action="store_true", dest="benchmark",
                      help="run all benchmarking tests", default=False)
    parser.add_option('-z', '--submit',
                      action="store", dest="submit",
                      help="submits performance data to VGL rating with specified arch name", default=None)

    options, args = parser.parse_args()

    run(options)


if __name__ == "__main__":
    main()


