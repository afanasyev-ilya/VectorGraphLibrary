import socket
import pickle
import sys
import socket
import ast
from pymongo import *
import datetime


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
                recv_data = conn.recv(1024)

                if not recv_data:
                    break

                conn.sendall("Server Says:results_accepted")
                recv_dict = pickle.loads(recv_data)
                print(recv_dict)

            except socket.error:
                print("Error Occured.")
                break

        conn.close()


listen_to_users(1026)
