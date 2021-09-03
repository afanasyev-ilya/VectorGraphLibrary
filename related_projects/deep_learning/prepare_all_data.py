import os
import optparse
import subprocess


def prepare_synthetic_learning_data():
    graph_types = ["rmat", "ru"]
    for graph_type in graph_types:
        for scale in range(15, 22):
            for edge_factor in range(5, 32):
                cmd = ["./generate_learning_data.out", "-s", str(scale), "-e", str(edge_factor), "-directed",
                       "-type", graph_type]
                print(' '.join(cmd))
                proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE)
                output = proc.stdout.read().decode("utf-8")


if __name__ == "__main__":
    prepare_synthetic_learning_data()




