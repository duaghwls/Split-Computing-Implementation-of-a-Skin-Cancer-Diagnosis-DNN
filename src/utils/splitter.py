"""
모델 분할 유틸리티
- Keras 모델을 특정 레이어를 기준으로 head와 tail로 분할
"""

import tensorflow as tf
from keras.models import Model


def split_model(model, split_layer_name):
    """
    원본 model과 분할하려는 layer의 이름을 입력받아 
    head와 tail 모델로 분할하여 반환
    
    Args:
        model: 분할할 원본 Keras 모델
        split_layer_name: 분할 기준이 될 레이어 이름
    
    Returns:
        head_model: 입력부터 split_layer까지의 모델
        tail_model: split_layer 다음부터 출력까지의 모델
        
    Raises:
        ValueError: split_layer_name에 해당하는 레이어를 찾을 수 없는 경우
    """
    # 분할할 레이어 및 분할 직후 레이어 탐색
    split_layer = None
    tail_input_layer = None
    
    for layer in model.layers:
        if split_layer is not None:
            tail_input_layer = layer
            break
        elif layer.name == split_layer_name:
            split_layer = layer
            continue
    
    # 레이어를 찾지 못한 경우 에러 처리
    if split_layer is None:
        available_layers = [layer.name for layer in model.layers]
        raise ValueError(
            f"'{split_layer_name}' 이름의 레이어를 모델에서 찾을 수 없습니다.\n"
            f"사용 가능한 레이어: {available_layers}"
        )
    
    if tail_input_layer is None:
        raise ValueError(
            f"'{split_layer_name}' 레이어가 모델의 마지막 레이어입니다. "
            f"분할할 수 없습니다."
        )
    
    # Head 모델 생성 (입력 ~ split_layer 출력)
    head_model = Model(
        inputs=model.input, 
        outputs=split_layer.output, 
        name="head_model"
    )
    
    # Tail 모델 생성 (tail_input_layer 입력 ~ 모델 출력)
    tail_model = Model(
        inputs=tail_input_layer.input, 
        outputs=model.output, 
        name="tail_model"
    )
    
    print(f"✓ 모델 분할 완료:")
    print(f"  - Head 모델: {model.input.shape} -> {split_layer.output.shape}")
    print(f"  - Tail 모델: {tail_input_layer.input.shape} -> {model.output.shape}")
    
    return head_model, tail_model


def print_model_split_info(model, split_layer_name):
    """
    모델 분할 정보를 출력하는 헬퍼 함수
    
    Args:
        model: 분석할 Keras 모델
        split_layer_name: 분할 기준 레이어 이름
    """
    print(f"\n{'='*60}")
    print(f"모델 분할 정보 - 기준 레이어: '{split_layer_name}'")
    print(f"{'='*60}")
    
    split_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == split_layer_name:
            split_idx = idx
            break
    
    if split_idx is None:
        print(f"❌ '{split_layer_name}' 레이어를 찾을 수 없습니다.")
        return
    
    print(f"\n[Head 부분] - {split_idx + 1}개 레이어")
    for idx, layer in enumerate(model.layers[:split_idx + 1]):
        marker = " ← 분할 지점" if idx == split_idx else ""
        print(f"  {idx:3d}. {layer.name:40s} {str(layer.output_shape):30s}{marker}")
    
    print(f"\n[Tail 부분] - {len(model.layers) - split_idx - 1}개 레이어")
    for idx, layer in enumerate(model.layers[split_idx + 1:], start=split_idx + 1):
        print(f"  {idx:3d}. {layer.name:40s} {str(layer.output_shape):30s}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 테스트 코드
    from keras.applications import InceptionResNetV2
    
    print("모델 분할 유틸리티 테스트\n")
    
    # InceptionResNetV2 모델 로드
    model = InceptionResNetV2(
        weights="imagenet", 
        include_top=False, 
        input_shape=(150, 150, 3)
    )
    
    split_layer = "block8_1"
    
    # 분할 정보 출력
    print_model_split_info(model, split_layer)
    
    # 모델 분할 테스트
    try:
        head, tail = split_model(model, split_layer)
        print("\n✓ 모델 분할 테스트 성공")
        print(f"\nHead 모델 요약:")
        head.summary()
        print(f"\nTail 모델 요약:")
        tail.summary()
    except Exception as e:
        print(f"\n❌ 모델 분할 테스트 실패: {e}")
