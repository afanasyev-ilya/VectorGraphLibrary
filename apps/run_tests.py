import os
import optparse
import scripts.settings
from scripts.benchmarking_api import *
from scripts.verification_api import *
from scripts.export_to_xls import BenchmarkingResults


def run_compile(options, arch):
    make_binary("clean", arch)
    list_of_apps = prepare_list_of_apps(options.apps)
    for app_name in list_of_apps:
        make_binary(app_name, arch)
        if not binary_exists(app_name, arch):
            print("Error! Can not compile " + app_name + ", several errors occurred.")


def run_prepare(options, arch):
    create_graphs_if_required(get_list_of_all_graphs(options.mode), arch, options.mode)


def run_benchmarks(options, arch, benchmarking_results):
    list_of_apps = prepare_list_of_apps(options.apps)

    set_omp_environments(options)
    benchmarking_results.add_performance_header_to_xls_table(options.format)

    algorithms_tested = 0
    for app_name in list_of_apps:
        if is_valid(app_name, arch, options):
            algorithms_tested += benchmark_app(app_name, arch, benchmarking_results, options.format, options.mode,
                                               options.timeout)
        else:
            print("Error! Can not benchmark " + app_name + ", several errors occurred.")
    return algorithms_tested


def run_verify(options, arch, benchmarking_results):
    list_of_apps = prepare_list_of_apps(options.apps)

    set_omp_environments(options)
    benchmarking_results.add_correctness_header_to_xls_table(options.format)
    algorithms_verified = 0

    for app_name in list_of_apps:
        if "el" in options.format or app_name == "sswp": # check takes too long to be done
            continue
        if is_valid(app_name, arch, options):
            algorithms_verified += verify_app(app_name, arch, benchmarking_results, options.format, options.mode,
                                              options.timeout)
        else:
            print("Error! Can not compile " + app_name + ", several errors occurred.")
    return algorithms_verified


def benchmark_and_verify(options, arch, benchmarking_results):
    benchmarked_num = 0
    verified_num = 0
    if options.benchmark:
        benchmarked_num = run_benchmarks(options, arch, benchmarking_results)

    if options.verify:
        verified_num = run_verify(options, arch, benchmarking_results)

    print("\n\nEVALUATED PERFORMANCE OF " + str(benchmarked_num) + " GRAPH ALGORITHMS\n")
    print("VERIFIED " + str(verified_num) + " GRAPH ALGORITHMS\n\n")

def run(options, run_info):
    create_dir("./bin/")
    arch = options.arch

    benchmarking_results = BenchmarkingResults(options.name, options.mode)

    if options.compile:
        start = time.time()
        run_compile(options, arch)
        end = time.time()
        if print_timings:
            print("compile WALL TIME: " + str(end-start) + " seconds")

    if options.download:
        start = time.time()
        download_all_real_world_graphs(options.mode)
        end = time.time()
        if print_timings:
            print("download WALL TIME: " + str(end-start) + " seconds")

    if options.prepare:
        start = time.time()
        run_prepare(options, arch)
        end = time.time()
        if print_timings:
            print("graph generation WALL TIME: " + str(end-start) + " seconds")

    start = time.time()
    if options.format == "all":
        for current_format in available_formats:
            options.format = current_format
            benchmark_and_verify(options, arch, benchmarking_results)
    elif "," in options.format:
        format_string = options.format
        format_list = format_string.split(",")
        for current_format in format_list:
            options.format = current_format
            benchmark_and_verify(options, arch, benchmarking_results)
    else:
        benchmark_and_verify(options, arch, benchmarking_results)
        if run_info != {}:
            run_info["format"] = options.format
            if benchmarking_results.submit(run_info):
                print("Results sent to server!")
                benchmarking_results.offline_submit(run_info, options.name)
            else:
                print("Can not send results, saving to file...")
                benchmarking_results.offline_submit(run_info, options.name)
    end = time.time()
    if print_timings:
        print("benchmarking WALL TIME: " + str(end-start) + " seconds")

    benchmarking_results.finalize()


def main():
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-a', '--apps',
                      action="store", dest="apps",
                      help="specify an application to test (or all for testing all available applications)", default="all")
    parser.add_option('-r', '--arch',
                      action="store", dest="arch",
                      help="specify evaluated architecture: sx/aurora, mc/multicore, cu/gpu", default=get_arch())
    parser.add_option('-f', '--formats',
                      action="store", dest="format",
                      help="specify graph storage format used (all can be specified to test all available formats)", default="vcsr")
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
    parser.add_option('-n', '--name',
                      action="store", dest="name",
                      help="specify name prefix of output files", default="unknown")
    parser.add_option('-m', '--mode',
                      action="store", dest="mode",
                      help="specify testing mode: tiny-only, small-only, medium-only, large-only"
                           "tiny-small, tiny-small-medium (combinations), "
                           "rating-full, rating-medium, rating-fast (for rating submission)", default="tiny-only")
    parser.add_option('-d', '--download',
                      action="store_true", dest="download",
                      help="download all real-world graphs from internet collections", default=False)
    parser.add_option('-t', '--timeout',
                      action="store", dest="timeout",
                      help="execution time (in seconds), after which tested app is automatically aborted. "
                           "Default is 1 hour", default=3600)

    options, args = parser.parse_args()

    run(options, {})


if __name__ == "__main__":
    main()


