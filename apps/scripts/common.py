from enum import Enum


benchmark_args = {"bfs": [ ["-top-down"], ["-do"] ],
                  "sssp": [ ["-push, -all-active"], ["-pull", "-all-active"], ["-push", "-partial-active"] ],
                  "pr": [ ["-push"], ["-pull"] ],
                  "cc": [ ["-bfs-based"], ["-cv"] ],
                  "sswp": [ ["-push"] ],
                  "rw": [ ["-store_walk_paths", "-walk-lengths", "32", "-walk-vertices_num", "100"],
                          ["-walk-lengths", "32", "-walk-vertices_num", "100"] ] }
common_iterations = 10
perf_pattern = "MAX_PERF"
correctness_pattern = "error count:"

GRAPHS_DIR = "./bin/input_graphs/"
SOURCE_GRAPH_DIR = "./bin/source_graphs/"
synthetic_min_scale = 18
synthetic_medium_scale = 24
synthetic_large_scale = 26
synthetic_edge_factor = 32


class BenchmarkMode(Enum):
    long = 1,
    medium = 2
    short = 3


mode = BenchmarkMode.medium
