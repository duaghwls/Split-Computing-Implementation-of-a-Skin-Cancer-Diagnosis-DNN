"""
클라이언트 추론 스크립트
- 이미지를 입력받아 Head 모델로 중간 특징 추출
- Socket을 통해 서버로 중간 결과 전송
- 서버로부터 최종 진단 결과 수신
"""

import os
import sys
import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras import layers, regularizers
from keras.applications import InceptionResNetV2

# 상위 디렉토리의 utils 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.splitter import split_model
from socket_client import send_to_server

# 설정값
MODEL_PATH = "../../models/SC_best_model_InceptionResNetV2.h5"
TEST_IMAGE_PATH = "../../assets/KakaoTalk_20230613_211026515.jpg"
IMAGE_SIZE = (150, 150)
SPLIT_LAYER = "block8_1"


def load_and_split_model(model_path, split_layer):
    """
    학습된 모델을 불러와서 head와 tail로 분할
    
    Args:
        model_path: 학습된 모델 파일 경로
        split_layer: 분할할 레이어 이름
    
    Returns:
        head: 클라이언트에서 실행할 head 모델
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
    head_model, tail_model = split_model(conv_base, split_layer)
    
    # Head 모델 구성
    head = Sequential()
    head.add(head_model)
    head.set_weights(trained_weight[0:730])
    
    # Tail 모델 구성 (참고용 - 실제로는 서버에서 사용)
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
    
    return head, tail


def preprocess_image(image_path, target_size):
    """
    이미지 전처리
    
    Args:
        image_path: 이미지 파일 경로
        target_size: 목표 이미지 크기 (width, height)
    
    Returns:
        전처리된 이미지 numpy array
    """
    if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("지원하지 않는 이미지 형식입니다. PNG, JPG, JPEG만 가능합니다.")
    
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image


def main():
    """메인 실행 함수"""
    try:
        print("=" * 50)
        print("피부암 진단 클라이언트 시작")
        print("=" * 50)
        
        # 모델 로드 및 분할
        print("\n[1/4] 모델 로드 및 분할 중...")
        head, _ = load_and_split_model(MODEL_PATH, SPLIT_LAYER)
        print("✓ 모델 로드 완료")
        
        # 이미지 전처리
        print("\n[2/4] 이미지 전처리 중...")
        if not os.path.exists(TEST_IMAGE_PATH):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {TEST_IMAGE_PATH}")
        
        image = preprocess_image(TEST_IMAGE_PATH, IMAGE_SIZE)
        print(f"✓ 이미지 로드 완료: {TEST_IMAGE_PATH}")
        
        # Head 모델로 중간 특징 추출
        print("\n[3/4] 클라이언트 추론 중...")
        intermediate_data = head.predict(image, verbose=0)
        print(f"✓ 중간 특징 추출 완료 (Shape: {intermediate_data.shape})")
        
        # 서버로 전송 및 결과 수신
        print("\n[4/4] 서버로 전송 중...")
        result_message = send_to_server(intermediate_data)
        
        # 결과 출력
        print("\n" + "=" * 50)
        print("진단 결과")
        print("=" * 50)
        print(f"이미지: {os.path.basename(TEST_IMAGE_PATH)}")
        print(f"결과: {result_message}")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("모델 파일 또는 이미지 파일의 경로를 확인해주세요.")
    except Exception as e:
        print(f"\n❌예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
