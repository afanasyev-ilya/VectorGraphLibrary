from pathlib import Path
import subprocess
import os
import shutil
from .settings import *
import urllib.request


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_binary_path(app_name, arch):
    return "./bin/" + app_name + get_binary_suffix(arch)


def get_binary_suffix(arch):
    suffix_table = {"sx": "_sx", "aurora": "_sx", "nec": "_sx",
                    "mc": "_mc", "multicore": "_mc", "cpu": "_mc",
                    "cu": "_cu", "gpu": "_cu",
                    "": ""}
    if arch not in suffix_table:
        error_val = 'Incorrect architecture ' + arch + "!"
        raise ValueError(error_val)
    return suffix_table[arch]


def get_compiler(arch):
    suffix_table = {"sx": "nc++", "aurora": "nc++", "nec": "nc++",
                    "mc": "g++", "multicore": "g++", "cpu": "g++",
                    "cu": "nvcc", "gpu": "nvcc"}
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
    if app_name == "clean":
        return True
    if file_exists(get_binary_path(app_name, arch)):
        return True
    print("Warning! path " + get_binary_path(app_name, arch) + " does not exist")
    return False


def make_binary(app_name, arch):
    cmd = "make " + app_name + " CXX=" + get_compiler(arch)
    print(cmd)
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if binary_exists(app_name, arch):
        print("Success! " + app_name + " has been compiled")
    else:
        print("Error! " + app_name + " can not be compiled")


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


def get_target_proc_model():  # returns number of sockets of target architecture
    try:
        output = subprocess.check_output(["lscpu"])
        model = "Unknown"
        for item in output.decode().split("\n"):
            if "Model name" in item:
                model_line = item.strip()
                model = int(model_line.split(":")[1])
        return model
    except:
        return "Unknown"


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


def internet_on():
    try:
        urllib.request.urlopen('http://216.58.192.142', timeout=1)
        return True
    except Exception as e:
        print(e)
        return False
