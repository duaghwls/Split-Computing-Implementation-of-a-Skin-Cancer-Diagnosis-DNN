import socket
import numpy as np
import pickle
import struct
import os

"""
socket 통신으로 client가 server에게 numpy array를 전송
"""


# Create a TCP/IP socket
def socket_client(numpy_array):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 환경변수 또는 localhost 사용 (실제 IP는 환경변수로 설정)
    localhost = os.getenv("SC_SERVER_HOST", "localhost")
    port = int(os.getenv("SC_SERVER_PORT", "3579"))
    
    # Connect the socket to the server's address and port
    server_address = (localhost, port)
    client_socket.connect(server_address)

    serialized_data = pickle.dumps(numpy_array)

    # 데이터 크기 전송
    data_size = len(serialized_data)
    client_socket.sendall(struct.pack("!I", data_size))

    print(data_size)

    client_socket.sendall(serialized_data)

    print("데이터가 전송되었습니다.\n")

    message = client_socket.recv(1024)
    message = message.decode()
    client_socket.close()

    return message
