import pickle
from scripts.submit_results import *

a_file = open("vgl_rating_data.pkl", "rb")
perf_stats = pickle.load(a_file)
submit(perf_stats)
a_file.close()