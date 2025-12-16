"""
ISIC 피부암 데이터셋 학습 스크립트
InceptionResNetV2 모델을 사용한 Split Computing용 모델 학습
"""

import os
import sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers, regularizers
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 상위 디렉토리의 utils 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.splitter import split_model

# ============================================================================
# 설정값
# ============================================================================

# 데이터 경로 (실제 환경에 맞게 수정 필요)
TRAIN_DIR = r"C:\project\ISIC_new\train"
VAL_DIR = r"C:\project\ISIC_new\validation"

# 모델 저장 경로
MODEL_SAVE_PATH = "../../models/SC_best_model_InceptionResNetV2.h5"

# 하이퍼파라미터
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.02
SPLIT_LAYER = "block8_1"

# ============================================================================
# GPU 설정
# ============================================================================

def setup_gpu():
    """GPU 사용 설정"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✓ GPU 사용: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("⚠ GPU를 찾을 수 없습니다. CPU를 사용합니다.")


# ============================================================================
# 데이터 로더
# ============================================================================

def create_data_generators(train_dir, val_dir, img_width, img_height, batch_size):
    """
    학습 및 검증 데이터 제너레이터 생성
    
    Args:
        train_dir: 학습 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리
        img_width: 이미지 너비
        img_height: 이미지 높이
        batch_size: 배치 크기
    
    Returns:
        train_generator: 학습 데이터 제너레이터
        val_generator: 검증 데이터 제너레이터
    """
    # 데이터 증강 설정 (학습 데이터)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # 검증 데이터 (증강 없음)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    # 데이터 로딩
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
    )
    
    print(f"\n✓ 데이터 로드 완료:")
    print(f"  - 학습 샘플: {train_generator.samples}개")
    print(f"  - 검증 샘플: {val_generator.samples}개")
    print(f"  - 클래스: {train_generator.class_indices}")
    
    return train_generator, val_generator


# ============================================================================
# 모델 생성
# ============================================================================

def create_split_model(img_width, img_height, split_layer, learning_rate):
    """
    Split Computing용 모델 생성
    
    Args:
        img_width: 이미지 너비
        img_height: 이미지 높이
        split_layer: 분할 레이어 이름
        learning_rate: 학습률
    
    Returns:
        model: 컴파일된 Keras 모델
    """
    # InceptionResNetV2 기본 모델 로드
    conv_base = InceptionResNetV2(
        weights="imagenet", 
        include_top=False, 
        input_shape=(img_width, img_height, 3)
    )
    
    # 모델 분할 (Split Computing 준비)
    head, tail = split_model(conv_base, split_layer)
    
    # 전체 모델 구성
    model = Sequential([
        head,
        tail,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.3),
        layers.Dense(2, activation="sigmoid")
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    print("\n✓ 모델 생성 완료")
    print(f"  - 기본 모델: InceptionResNetV2")
    print(f"  - 분할 레이어: {split_layer}")
    print(f"  - 학습률: {learning_rate}")
    
    return model


# ============================================================================
# 콜백 설정
# ============================================================================

def create_callbacks(model_save_path):
    """
    학습 콜백 생성
    
    Args:
        model_save_path: 모델 저장 경로
    
    Returns:
        callbacks: 콜백 리스트
    """
    # 모델 저장 디렉토리 생성
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        # 최고 성능 모델 저장
        ModelCheckpoint(
            model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        # Early Stopping
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 학습률 감소
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 학습 함수"""
    print("=" * 70)
    print("ISIC 피부암 진단 모델 학습")
    print("=" * 70)
    
    # GPU 설정
    setup_gpu()
    
    # 데이터 로드
    print("\n[1/4] 데이터 로딩 중...")
    train_gen, val_gen = create_data_generators(
        TRAIN_DIR, VAL_DIR, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
    )
    
    # 분할 전략 (Multi-GPU 지원)
    strategy = tf.distribute.MirroredStrategy()
    print(f"\n사용 가능한 장치 수: {strategy.num_replicas_in_sync}")
    
    with strategy.scope():
        # 모델 생성
        print("\n[2/4] 모델 생성 중...")
        model = create_split_model(
            IMG_WIDTH, IMG_HEIGHT, SPLIT_LAYER, LEARNING_RATE
        )
        
        # 모델 요약
        print("\n모델 구조:")
        model.summary()
        
        # 콜백 설정
        print("\n[3/4] 콜백 설정 중...")
        callbacks = create_callbacks(MODEL_SAVE_PATH)
        
        # 모델 학습
        print("\n[4/4] 학습 시작...")
        print("=" * 70)
        
        history = model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=val_gen.samples // val_gen.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "=" * 70)
        print("✓ 학습 완료!")
        print(f"모델 저장 위치: {MODEL_SAVE_PATH}")
        print("=" * 70)
        
        # 최종 성능 출력
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\n최종 학습 정확도: {final_train_acc:.4f}")
        print(f"최종 검증 정확도: {final_val_acc:.4f}")
        
        return history


if __name__ == "__main__":
    try:
        history = main()
    except KeyboardInterrupt:
        print("\n\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
