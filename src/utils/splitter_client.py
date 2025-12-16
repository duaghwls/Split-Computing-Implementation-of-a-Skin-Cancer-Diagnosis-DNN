import tensorflow as tf

"""
원본 model과 분할하려는 layer의 이름을 input으로 받아 head와 tail 모델로 분할하여 reurn
"""


def split_model(model, split_layer_name):
    # InceptionResNetV2 모델 로드
    base_model = model

    # 분할할 레이어 및 분할 직후 레이어 탐색
    split_layer = None
    tail_input_layer = None
    for layer in base_model.layers:
        if split_layer is not None:
            tail_input_layer = layer
            break
        elif layer.name == split_layer_name:
            split_layer = layer
            continue

    if split_layer is None:
        print(f"No layer with name {split_layer_name} found in the model.")
        return None, None

    # head 모델 생성
    head_model = tf.keras.Model(
        inputs=base_model.input, outputs=split_layer.output, name="head_model"
    )

    # tail 모델 생성
    tail_model = tf.keras.Model(
        inputs=tail_input_layer.input, outputs=base_model.output, name="tail_model"
    )

    return head_model, tail_model


# 예시 사용
# head_model, tail_model = split_model('split_layer_name')
