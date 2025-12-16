"""
Socket 클라이언트 모듈
- 서버에 numpy array 데이터를 전송
- 서버로부터 추론 결과를 수신
"""

import os
import sys
import socket
import numpy as np
import pickle
import struct

# config 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import SERVER_HOST, SERVER_PORT


def send_to_server(numpy_array, host=SERVER_HOST, port=SERVER_PORT):
    """
    Socket 통신으로 서버에 numpy array 전송하고 결과 수신
    
    Args:
        numpy_array: 서버로 전송할 numpy array (중간 특징 데이터)
        host: 서버 호스트 주소
        port: 서버 포트 번호
    
    Returns:
        서버로부터 받은 추론 결과 메시지
    """
    client_socket = None
    
    try:
        # TCP/IP 소켓 생성
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 서버 연결
        server_address = (host, port)
        print(f"서버 연결 중... ({host}:{port})")
        client_socket.connect(server_address)
        print("✓ 서버 연결 성공")
        
        # 데이터 직렬화
        serialized_data = pickle.dumps(numpy_array)
        data_size = len(serialized_data)
        
        # 데이터 크기 전송 (4바이트)
        client_socket.sendall(struct.pack("!I", data_size))
        print(f"데이터 크기 전송: {data_size:,} bytes")
        
        # 실제 데이터 전송
        client_socket.sendall(serialized_data)
        print("✓ 데이터 전송 완료")
        
        # 서버로부터 결과 수신
        result_message = client_socket.recv(1024)
        result_message = result_message.decode('utf-8')
        
        return result_message
        
    except socket.error as e:
        print(f"❌ 소켓 오류: {e}")
        raise
    except Exception as e:
        print(f"❌ 전송 중 오류 발생: {e}")
        raise
    finally:
        # 소켓 닫기
        if client_socket:
            client_socket.close()
            print("✓ 연결 종료")


if __name__ == "__main__":
    # 테스트용 코드
    print("Socket 클라이언트 모듈 테스트")
    test_data = np.random.rand(1, 4, 4, 1536)  # 임의의 테스트 데이터
    print(f"테스트 데이터 Shape: {test_data.shape}")
    
    try:
        result = send_to_server(test_data)
        print(f"서버 응답: {result}")
    except Exception as e:
        print(f"테스트 실패: {e}")
