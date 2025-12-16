# Split Computing 기반 피부암 진단 시스템 - 사용 가이드

## 📚 목차

1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [시스템 실행](#시스템-실행)
5. [문제 해결](#문제-해결)

## 🔧 환경 설정

### 1. Python 환경 준비

```bash
# Python 3.8 이상 필요
python --version

# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 의존성 설치

```bash
# requirements.txt를 이용한 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install tensorflow keras numpy pillow opencv-python matplotlib scikit-learn
```

### 3. 설정 파일 수정

`src/config.py` 파일을 열어 환경에 맞게 수정:

```python
# 서버 IP 주소 및 포트
SERVER_HOST = "192.168.0.3"  # 실제 서버 IP로 변경
SERVER_PORT = 3999

# 데이터 경로
TRAIN_DATA_DIR = "C:/project/ISIC_new/train"  # 실제 경로로 변경
VAL_DATA_DIR = "C:/project/ISIC_new/validation"
```

## 📊 데이터 준비

### ISIC 데이터셋 다운로드

1. [ISIC Archive](https://www.isic-archive.com/) 접속
2. 데이터셋 다운로드 (권장: ISIC 2019 또는 2020)
3. 다음 구조로 데이터 배치:

```
data/
├── train/
│   ├── benign/     # 양성 종양 이미지
│   └── malignant/  # 악성 종양 이미지
├── validation/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

### 데이터 전처리

```bash
# (선택사항) 데이터 증강 및 전처리
python src/utils/preprocess_data.py
```

## 🎓 모델 학습

### 1. 학습 실행

```bash
cd src/training
python train_model.py
```

### 2. 학습 옵션 조정

`src/config.py`에서 하이퍼파라미터 조정:

```python
BATCH_SIZE = 64        # 배치 크기
EPOCHS = 20           # 에폭 수
LEARNING_RATE = 0.02  # 학습률
```

### 3. 학습 모니터링

학습 중 다음 정보가 출력됩니다:
- Epoch별 학습/검증 정확도
- 손실 값
- 최고 성능 모델 자동 저장

### 4. 학습 결과

학습이 완료되면 `models/` 폴더에 다음 파일이 생성됩니다:
- `SC_best_model_InceptionResNetV2.h5` - 최고 성능 모델

## 🚀 시스템 실행

### 1. 서버 실행 (먼저 실행)

```bash
cd src/server
python server_inference.py
```

출력 예시:
```
==================================================
피부암 진단 서버 시작
==================================================
주소: 192.168.0.3:3999
클라이언트 연결 대기 중...
```

### 2. 클라이언트 실행 (별도 터미널)

```bash
cd src/client
python client_inference.py
```

출력 예시:
```
==================================================
피부암 진단 클라이언트 시작
==================================================

[1/4] 모델 로드 및 분할 중...
✓ 모델 로드 완료

[2/4] 이미지 전처리 중...
✓ 이미지 로드 완료: test_image.jpg

[3/4] 클라이언트 추론 중...
✓ 중간 특징 추출 완료

[4/4] 서버로 전송 중...
✓ 결과 수신 완료

==================================================
진단 결과
==================================================
이미지: test_image.jpg
결과: 해당 이미지는 정상입니다. (악성 확률: 12.34%)
==================================================
```

## ⚙️ 고급 사용법

### 커스텀 이미지로 테스트

`client_inference.py` 수정:

```python
TEST_IMAGE_PATH = "../../assets/my_test_image.jpg"  # 테스트할 이미지 경로
```

### 다른 레이어에서 분할

`src/config.py` 수정:

```python
SPLIT_LAYER = "block8_2"  # 다른 레이어로 변경
```

사용 가능한 레이어 확인:

```bash
cd src/utils
python splitter.py
```

### 임계값 조정

`src/config.py`에서 악성 판정 임계값 조정:

```python
MALIGNANT_THRESHOLD = 0.85  # 0.9에서 0.85로 변경 (더 민감하게)
```

## 🔍 문제 해결

### 모델 파일을 찾을 수 없음

```
❌ 오류: 모델 파일을 찾을 수 없습니다
```

**해결방법**: 
1. `models/` 폴더 확인
2. 모델 학습이 완료되었는지 확인
3. `src/config.py`의 `MODEL_PATH` 확인

### 서버 연결 실패

```
❌ 소켓 오류: [Errno 111] Connection refused
```

**해결방법**:
1. 서버가 먼저 실행되었는지 확인
2. `SERVER_HOST`와 `SERVER_PORT` 확인
3. 방화벽 설정 확인

### GPU 메모리 부족

```
ResourceExhaustedError: OOM when allocating tensor
```

**해결방법**:
1. 배치 크기 줄이기: `BATCH_SIZE = 32`
2. 이미지 크기 줄이기: `IMAGE_WIDTH = IMAGE_HEIGHT = 128`

### 데이터셋 로드 오류

```
Found 0 images belonging to 0 classes
```

**해결방법**:
1. 데이터 경로 확인
2. 폴더 구조 확인 (train/benign, train/malignant 등)
3. 이미지 파일 형식 확인 (jpg, png 등)

## 📞 추가 도움말

더 자세한 내용은 다음을 참조하세요:
- `docs/중간발표_script.docx` - 프로젝트 설명
- `docs/종합설계 포스터.pdf` - 시스템 아키텍처
- `README.md` - 프로젝트 개요

## 🐛 버그 리포트

문제가 해결되지 않으면:
1. 에러 메시지 전체 복사
2. 실행 환경 정보 (OS, Python 버전 등)
3. GitHub Issues에 등록
