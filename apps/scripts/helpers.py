from pathlib import Path
import subprocess
import os
import shutil
from .common import *


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_binary_path(app_name, arch):
    return "./bin/" + app_name + get_binary_suffix(arch)


def get_binary_suffix(arch):
    suffix_table = {"sx": "_sx", "aurora": "_sx",
                    "mc": "_mc", "multicore": "_mc", "cpu": "_mc",
                    "cu": "_cu", "gpu": "_cu",
                    "": ""}
    if arch not in suffix_table:
        error_val = 'Incorrect architecture ' + arch + "!"
        raise ValueError(error_val)
    return suffix_table[arch]


def get_makefile_suffix(arch):
    suffix_table = {"sx": "nec", "aurora": "nec",
                    "mc": "gcc", "multicore": "gcc", "cpu": "gcc",
                    "cu": "cu", "gpu": "cu"}
    if arch not in suffix_table:
        error_val = 'Incorrect architecture ' + arch + "!"
        raise ValueError(error_val)
    return suffix_table[arch]


def file_exists(path):
    bin_path = Path(path)
    if bin_path.is_file():
        return True
    return False


def binary_exists(app_name, arch):
    if file_exists(get_binary_path(app_name, arch)):
        return True
    print("Warning! path " + get_binary_path(app_name, arch) + " does not exist")
    return False


def make_binary(app_name, arch):
    cmd = "make -f Makefile." + get_makefile_suffix(arch) + " " + app_name
    print(cmd)
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def is_valid(app_name, arch, options):
    if not binary_exists(app_name, arch):
        make_binary(app_name, arch)
        if not binary_exists(app_name, arch):
            return False
    return True


def get_cores_count():  # returns number of sockets of target architecture
    try:
        output = subprocess.check_output(["lscpu"])
        cores = -1
        for item in output.decode().split("\n"):
            if "Core(s) per socket:" in item:
                cores_line = item.strip()
                cores = int(cores_line.split(":")[1])
        if cores == -1:
            raise NameError('Can not detect number of cores of target architecture')
        return cores
    except:
        cores = 8 # SX-Aurora
        return cores


def get_sockets_count():  # returns number of sockets of target architecture
    try:
        output = subprocess.check_output(["lscpu"])
        cores = -1
        sockets = -1
        for item in output.decode().split("\n"):
            if "Socket(s)" in item:
                sockets_line = item.strip()
                sockets = int(sockets_line.split(":")[1])
        if sockets == -1:
            raise NameError('Can not detect number of cores of target architecture')
        return sockets
    except:
        sockets = 1 # SX-Aurora
        return sockets


def get_threads_count():
    return get_sockets_count()*get_cores_count()


def set_omp_environments(options):
    threads = get_cores_count()
    if int(options.sockets) > 1:
        threads = int(options.sockets) * threads
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OMP_PROC_BIND'] = 'true'
    os.environ['OMP_PROC_BIND'] = 'close'


def prepare_list_of_apps(apps_string):
    if apps_string == "all":
        apps_list = []
        for app in benchmark_args:
            apps_list += [app]
        return apps_list
    else:
        return apps_string.split(",")
