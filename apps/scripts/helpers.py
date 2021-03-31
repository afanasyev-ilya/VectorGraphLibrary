from pathlib import Path
import subprocess
import os
import shutil


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
    print("Warning: path " + get_binary_path(app_name, arch) + " does not exist")
    return False


def make_binary(app_name, arch):
    cmd = "make -f Makefile." + get_makefile_suffix(arch) + " " + app_name
    print(cmd)
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).wait()


def is_valid(app_name, arch):
    if not binary_exists(app_name, arch):
        make_binary(app_name, arch)
        if not binary_exists(app_name, arch):
            return False
    return True
