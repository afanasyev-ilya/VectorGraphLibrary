from enum import Enum


benchmark_args = {"bfs": [["-top-down"], ["-do"]],
                  "sssp": [["-push, -all-active"], ["-pull", "-all-active"], ["-push", "-partial-active"]],
                  "pr": [["-push"], ["-pull"]]}
common_iterations = 10
perf_pattern = "MAX_PERF"
correctness_pattern = "error count:"

GRAPHS_DIR = "./bin/input_graphs/"
SOURCE_GRAPH_DIR = "./bin/source_graphs/"
min_scale = 18


class BenchmarkMode(Enum):
    long = 1,
    short = 2


mode = BenchmarkMode.short