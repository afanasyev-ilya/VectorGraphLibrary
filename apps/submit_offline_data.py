import pickle
from scripts.submit_results import *
import sys

a_file = open("vgl_rating_data.pkl", "rb")
perf_stats = pickle.load(a_file)
submit(perf_stats)
a_file.close()


def select_mode_and_run():
    if len(sys.argv) == 1:
        print("test")
