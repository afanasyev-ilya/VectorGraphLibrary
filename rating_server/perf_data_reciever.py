import pickle
import socket
import struct
from pymongo import *


def verify_correctness(correctness_data):
    return True


def connect_to_mongo():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['vgl_rankings_db']
        perf_data_collection = db['perf_data_collection']
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)


def check_if_results_for_arch_exist():
    return True


def add_item():
    return True


def process_results(benchmarking_data):
    arch_name = benchmarking_data["arch"]
    perf_data = benchmarking_data["perf_data"]
    correctness_data = benchmarking_data["correctness_data"]
    print(arch_name)
    print(perf_data)
    print(correctness_data)

    correctness = verify_correctness(correctness_data)

    connect_to_mongo()

    # if arch exists - add
    # if arch does not exist - update if better

    return correctness


def recvall(sock):
    BUFF_SIZE = 4096 # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            break
    return data


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

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.bind((host, port))
    print(str(host) + str(port))

    while True:
        s.listen(1)
        conn, addr = s.accept()
        print('Connected by', addr)
        while True:

            try:
                recv_data = conn.recv(16*1024)
                if not recv_data:
                    break

                benchmarking_data = pickle.loads(recv_data)
                print(benchmarking_data)

                response = "accepted"
                conn.sendall(response.encode())
            except socket.error:
                print("Error Occured.")
                break

        conn.close()


listen_to_users(1026)
