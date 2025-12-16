"""
서버 추론 스크립트
- 클라이언트로부터 중간 특징 데이터 수신
- Tail 모델로 최종 추론 수행
- 결과를 클라이언트에 전송
"""

import os
import sys
import socket
import pickle
import numpy as np
import struct
from keras.models import Sequential, load_model
from keras import layers, regularizers
from keras.applications import InceptionResNetV2

# 상위 디렉토리의 utils 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.splitter import split_model
from config import SERVER_HOST, SERVER_PORT, MALIGNANT_THRESHOLD as THRESHOLD

# 설정값
MODEL_PATH = "../../models/SC_best_model_InceptionResNetV2.h5"
SPLIT_LAYER = "block8_1"


def load_tail_model(model_path, split_layer):
    """
    학습된 모델을 불러와서 tail 모델 생성
    
    Args:
        model_path: 학습된 모델 파일 경로
        split_layer: 분할할 레이어 이름
    
    Returns:
        tail: 서버에서 실행할 tail 모델
    """
    # 학습된 모델 불러오기
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    trained_model = load_model(model_path)
    trained_weight = trained_model.get_weights()
    
    # InceptionResNetV2 기반 모델 생성 및 분할
    conv_base = InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=(150, 150, 3)
    )
    _, tail_model = split_model(conv_base, split_layer)
    
    # Tail 모델 구성
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
    
    return tail


def receive_from_client(client_socket):
    """
    클라이언트로부터 데이터 수신
    
    Args:
        client_socket: 클라이언트 소켓
    
    Returns:
        수신된 numpy array 데이터
    """
    # 데이터 크기 수신
    data_size_buffer = client_socket.recv(4)
    data_size = struct.unpack("!I", data_size_buffer)[0]
    print(f"수신 예정 데이터 크기: {data_size:,} bytes")
    
    # 데이터 수신
    data_buffer = b""
    received_bytes = 0
    
    while len(data_buffer) < data_size:
        chunk = client_socket.recv(min(4096, data_size - len(data_buffer)))
        if not chunk:
            break
        data_buffer += chunk
        received_bytes = len(data_buffer)
        
        # 진행률 표시
        progress = (received_bytes / data_size) * 100
        print(f"수신 중... {received_bytes:,}/{data_size:,} bytes ({progress:.1f}%)", end='\r')
    
    print()  # 줄바꿈
    
    # 데이터 역직렬화
    received_array = pickle.loads(data_buffer)
    print(f"✓ 데이터 수신 완료 (Shape: {received_array.shape})")
    
    return received_array


def classify_result(prediction, threshold=THRESHOLD):
    """
    예측 결과를 해석하여 메시지 생성
    
    Args:
        prediction: 모델 예측 결과 (2D array)
        threshold: 악성 판정 임계값
    
    Returns:
        진단 결과 메시지
    """
    malignant_prob = prediction[0][1]  # 악성 확률
    benign_prob = prediction[0][0]     # 양성 확률
    
    print(f"\n예측 확률: 양성={benign_prob:.4f}, 악성={malignant_prob:.4f}")
    
    if malignant_prob > threshold:
        message = f"해당 이미지는 악성 종양일 가능성이 있습니다. (확률: {malignant_prob*100:.2f}%)"
    else:
        message = f"해당 이미지는 정상입니다. (악성 확률: {malignant_prob*100:.2f}%)"
    
    return message


def run_server(tail_model, host=SERVER_HOST, port=SERVER_PORT):
    """
    서버 실행 - 클라이언트 연결 대기 및 처리
    
    Args:
        tail_model: 추론에 사용할 tail 모델
        host: 서버 호스트 주소
        port: 서버 포트 번호
    """
    print("=" * 60)
    print("피부암 진단 서버 시작")
    print("=" * 60)
    print(f"주소: {host}:{port}")
    print("클라이언트 연결 대기 중...\n")
    
    request_count = 0
    
    while True:
        server_socket = None
        client_socket = None
        
        try:
            # 서버 소켓 생성 및 바인딩
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(1)
            
            # 클라이언트 연결 대기
            client_socket, addr = server_socket.accept()
            request_count += 1
            
            print(f"\n{'='*60}")
            print(f"요청 #{request_count}")
            print(f"클라이언트 연결됨: {addr}")
            print(f"{'='*60}")
            
            # 데이터 수신
            intermediate_data = receive_from_client(client_socket)
            
            # Tail 모델로 추론 수행
            print("\n추론 중...")
            inference_result = tail_model.predict(intermediate_data, verbose=0)
            print("✓ 추론 완료")
            
            # 결과 해석
            result_message = classify_result(inference_result)
            
            # 클라이언트에게 결과 전송
            client_socket.sendall(result_message.encode('utf-8'))
            print(f"✓ 결과 전송 완료: {result_message}")
            
        except KeyboardInterrupt:
            print("\n\n서버 종료 중...")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 소켓 정리
            if client_socket:
                client_socket.close()
            if server_socket:
                server_socket.close()
            print(f"\n연결 종료")
            print(f"{'='*60}\n")
            print("다음 클라이언트 연결 대기 중...\n")


def main():
    """메인 실행 함수"""
    try:
        print("모델 로드 중...")
        tail_model = load_tail_model(MODEL_PATH, SPLIT_LAYER)
        print("✓ 모델 로드 완료\n")
        
        # 서버 실행
        run_server(tail_model)
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("모델 파일의 경로를 확인해주세요.")
    except Exception as e:
        print(f"\n❌ 서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
