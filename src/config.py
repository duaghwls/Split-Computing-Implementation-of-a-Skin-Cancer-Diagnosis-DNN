"""
프로젝트 설정 파일
전역 설정값을 관리합니다.
"""

import os

# ============================================================================
# 경로 설정
# ============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 디렉토리
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(DATA_DIR, "validation")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

# 모델 디렉토리
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "SC_best_model_InceptionResNetV2.h5")

# 에셋 디렉토리
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# ============================================================================
# 모델 설정
# ============================================================================

# 이미지 설정
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3

# 모델 설정
BASE_MODEL = "InceptionResNetV2"
SPLIT_LAYER = "block8_1"
NUM_CLASSES = 2

# 학습 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.02

# Dropout 및 정규화
DROPOUT_RATE = 0.3
L2_REGULARIZATION = 0.005

# ============================================================================
# 네트워크 설정
# ============================================================================

# 서버 설정
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")  # 환경변수 또는 localhost
SERVER_PORT = int(os.getenv("SERVER_PORT", "3999"))

# 클라이언트 설정
CLIENT_TIMEOUT = 30  # 초

# Socket 설정
SOCKET_BUFFER_SIZE = 4096

# ============================================================================
# 추론 설정
# ============================================================================

# 악성 판정 임계값 (확률)
MALIGNANT_THRESHOLD = 0.9

# ============================================================================
# 로깅 설정
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
