import socket, pickle, struct

HOST = 'vgl-rating.parallel.ru'
PORT = 1026


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


def submit_to_socket(send_dict):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    success = True

    try:
        client_socket.connect((HOST, PORT))
    except:
        print("Can not connect to server...")
        return False

    send_data = pickle.dumps(send_dict)

    #send_msg(client_socket, send_string)
    client_socket.sendall(send_data)

    #response = recv_msg(client_socket)

    response = client_socket.recv(4096)
    response_str = response.decode()
    print("response: " + response_str)
    if "accepted" != response_str:
        success = False

    client_socket.close()
    return success
