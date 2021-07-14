import pickle
import socket
import struct
from mongo_api import *


def verify_correctness(correctness_data):
    number_of_errors = 0
    for graph_correctness in correctness_data:
        print(graph_correctness)
        # TODO
    return True


def process_results(benchmarking_data):
    arch_name = benchmarking_data["arch"]
    performance_data = benchmarking_data["performance_data"]
    correctness_data = benchmarking_data["correctness_data"]
    print(arch_name)
    print(performance_data)
    print(correctness_data)

    correctness = verify_correctness(correctness_data)

    if check_if_results_for_arch_exist(arch_name):
        add_performance_stats(performance_data, arch_name)

    return correctness


def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return sock.recv(msglen)


def listen_to_users(port_name):
    host = ''        # Symbolic name meaning all available interfaces
    port = port_name     # Arbitrary non-privileged port

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((host, port))

    dump_db_data()
    remove_collection()

    while True:
        server_sock.listen(1)
        conn, addr = server_sock.accept()
        print('Connected by', addr)
        while True:

            try:
                recv_data = conn.recv(4024*4)
                if not recv_data:
                    break

                benchmarking_data = pickle.loads(recv_data)
                if process_results(benchmarking_data):
                    response = "accepted"
                else:
                    response = "NOT accepted"

                conn.sendall(response.encode())
            except socket.error:
                print("Error Occured.")
                break

        conn.close()


listen_to_users(1026)
