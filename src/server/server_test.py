import socket
import pickle
import numpy as np
import struct
import os
from PIL import Image
from keras.models import Sequential, load_model
from keras import layers, regularizers
from keras.applications import InceptionResNetV2
import splitter

# 모델 불러오기
trained_model = load_model("SC_best_model_InceptionResNetV2.h5")
trained_weight = trained_model.get_weights()

# 모델 불러오기 및 분할
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
head_model, tail_model = splitter.split_model(conv_base, "block8_1")

head = Sequential()
head.add(head_model)
head.set_weights(trained_weight[0:730])

tail = Sequential()
tail.add(tail_model)
tail.add(layers.GlobalAveragePooling2D())
tail.add(layers.Dropout(0.3))
tail.add(
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.005))
)
tail.add(layers.Dropout(0.3))
tail.add(layers.Dense(2, activation="sigmoid"))
tail.set_weights(trained_weight[730:])


while True:
    # socket으로 intermediate_data 수신
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 환경변수 또는 localhost 사용
    localhost = os.getenv("SERVER_HOST", "localhost")
    port = int(os.getenv("SERVER_PORT", "3999"))
    
    server_socket.bind((localhost, port))

    server_socket.listen(1)
    print("서버가 시작되었습니다. 연결을 기다리는 중...")

    client_socket, addr = server_socket.accept()
    print(f"클라이언트가 연결되었습니다. {addr}")

    # 클라이언트로부터 데이터 사이즈 수신
    data_size_buffer = client_socket.recv(4)
    data_size = struct.unpack("!I", data_size_buffer)[0]

    data_buffer = b""
    while len(data_buffer) < data_size:
        recv_data = client_socket.recv(1024)
        if not recv_data:
            break
        data_buffer += recv_data
        print(len(data_buffer))

    # 수신된 데이터 역직렬화
    intermediat_data = pickle.loads(data_buffer)

    # tail inference 수행 예측 및 client에게 전송
    inference_result = tail.predict(intermediat_data)

    if inference_result[0][1] > 0.9:
        client_socket.sendall("해당 이미지는 악성 종양일 가능성이 있습니다.".encode())
    else:
        client_socket.sendall("해당 이미지는 정상입니다.".encode())

    client_socket.close()

    server_socket.close()
