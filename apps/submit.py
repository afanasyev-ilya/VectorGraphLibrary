import pickle
from scripts.submit_results import *
import optparse
from run_tests import run
import ast


def parse_dict(file_name):
    file = open(file_name, "r")

    contents = file.read()
    dictionary = ast.literal_eval(contents)

    file.close()

    if "architecture" not in dictionary:
        raise ValueError('architecture is not specified in target dictionary.')
    if "model" not in dictionary:
        raise ValueError('model is not specified in target dictionary.')
    if "author" not in dictionary:
        raise ValueError('author is not specified in target dictionary.')
    return dictionary


def main():
    parser = optparse.OptionParser()
    parser.add_option('-a', '--arch',
                      action="store", dest="arch",
                      help="specify evaluated architecture: sx/aurora, mc/multicore, cu/gpu", default="mc")
    parser.add_option('-f', '--format',
                      action="store", dest="format",
                      help="specify graph storage format used", default="vcsr")
    parser.add_option('-p', '--proc-num',
                      action="store", dest="proc_num",
                      help="set number of sockets / gpu / vector engines used (in testing)", default=1)
    parser.add_option('-o', '--offline',
                      action="store", dest="file_name",
                      help="specify file name for offline submit, other options are ignored this way", default=None)
    parser.add_option('-n', '--name',
                      action="store", dest="sub_name",
                      help="specify name of file with submission info", default="test")

    options, args = parser.parse_args()

    arch_info_dict = parse_dict(options.sub_name)

    if options.file_name is None:
        print("Doing online submit...")
        options.apps = 'bfs,sssp,pr,hits'
        options.verify = True
        options.benchmark = True
        options.compile = True
        options.prepare = False
        options.sockets = options.proc_num
        run(options, arch_info_dict)
    else:
        print("Doing offline submit...")
        a_file = open(options.file_name, "rb")
        perf_stats = pickle.load(a_file)
        submit_to_socket(perf_stats)
        a_file.close()


if __name__ == "__main__":
    main()
