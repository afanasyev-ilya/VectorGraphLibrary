import pickle
import socket
import struct
from web_rating.lib import mongo_api
from web_rating.lib.meta_data import add_meta_data
import json


def verify_correctness(correctness_data):
    number_of_errors = 0
    for graph_correctness in correctness_data:
        print(graph_correctness)
        # TODO
    return True


def arch_is_already_in_db(arch_name):
    if mongo_api.count_documents({"arch_name": arch_name}) > 0:
        return True
    return False


def insert_data_into_db(received_data, arch, arch_dict):
    for i in range(len(received_data)):
        print("inserting new")
        received_data[i] = add_meta_data(received_data[i], arch, arch_dict)

    for received_document in received_data:
        print(received_document)

    mongo_api.insert_many(received_data)


def update_data_in_db(received_data, arch, arch_dict):
    for new_val in received_data:
        search_pattern = {"graph_name": new_val["graph_name"], "arch_name": arch, "app_name": new_val["app_name"]}#, "arch_dict": arch_dict}
        old_val = mongo_api.find(search_pattern)

        if len(old_val) > 0:
            print(str(old_val[0]["perf_val"]) + " vs " + str(new_val["perf_val"]))
            new_perf = new_val["perf_val"]
            old_perf = old_val[0]["perf_val"]
            mongo_api.update_perf(search_pattern, max(new_perf, old_perf))
        else:
            new_val = add_meta_data(new_val, arch, arch_dict)
            mongo_api.insert_many([new_val])


def process_results(benchmarking_data):
    #print("arch_name = json.dumps( ", benchmarking_data["run_info"], " )")
    #print("--------------------------------------------------------------------\n")
    arch_name = json.dumps(benchmarking_data["run_info"])
    arch_dict = benchmarking_data["run_info"]
    performance_data = benchmarking_data["performance_data"]
    correctness_data = benchmarking_data["correctness_data"]
    correctness = verify_correctness(correctness_data) # TODO

    if arch_is_already_in_db(arch_name):
        update_data_in_db(performance_data, arch_name, arch_dict)
    else:
        insert_data_into_db(performance_data, arch_name, arch_dict)

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

    #dump_db_data()
    #remove_collection()

    NoErrors = True

    while NoErrors:
        server_sock.listen(1)
        conn, addr = server_sock.accept()
        print('Connected by', addr)
        while NoErrors:
            try:
                recv_data = conn.recv(4024*4)
                if not recv_data:
                    response = "Not Accepted. Connection Error."
                    NoErrors = False
                    break;
                else:
                    benchmarking_data = pickle.loads(recv_data)
                    if process_results(benchmarking_data):
                        response = "Accepted"
                    else:
                        response = "Not Accepted. Load Error."
                        NoErrors = False
                        break;

                conn.sendall(response.encode())
            except socket.error:
                response = "Not Accepted. Socket Error."
                NoErrors = False;
                break;

        conn.close()


listen_to_users(1026)
