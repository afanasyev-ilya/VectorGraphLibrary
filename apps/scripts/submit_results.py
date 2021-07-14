import socket, pickle

HOST = '100.64.127.244'
PORT = 5789


def submit(arch_name, perf_data, correctness_data):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    send_dict = {"arch": arch_name, "perf_data": perf_data, "correctness_data": correctness_data}

    client_socket.connect((HOST, PORT))
    send_string = pickle.dumps(send_dict)
    client_socket.send(send_string)

    response = client_socket.recv(1024)
    print(response.decode())