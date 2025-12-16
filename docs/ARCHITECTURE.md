# 프로젝트 아키텍처

## 시스템 개요

본 프로젝트는 Split Computing 기술을 활용하여 딥러닝 모델의 추론을 클라이언트-서버 환경에서 분할 처리합니다.

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                         전체 시스템 구조                          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐                           ┌─────────────────┐
│   Client Side    │                           │   Server Side   │
│                  │                           │                 │
│  ┌────────────┐  │                           │  ┌───────────┐  │
│  │   Image    │  │                           │  │   Tail    │  │
│  │   Input    │  │                           │  │   Model   │  │
│  └─────┬──────┘  │                           │  └─────┬─────┘  │
│        │         │                           │        │        │
│        ▼         │                           │        │        │
│  ┌────────────┐  │     Socket Transfer       │        │        │
│  │   Head     │  │    ─────────────────►     │        ▼        │
│  │   Model    │  │   Intermediate Features   │  ┌───────────┐  │
│  └─────┬──────┘  │                           │  │ Inference │  │
│        │         │                           │  └─────┬─────┘  │
│        ▼         │                           │        │        │
│  ┌────────────┐  │                           │        │        │
│  │  Feature   │  │    ◄─────────────────     │        ▼        │
│  │  Extract   │  │      Result Message       │  ┌───────────┐  │
│  └────────────┘  │                           │  │  Result   │  │
│                  │                           │  └───────────┘  │
└──────────────────┘                           └─────────────────┘
```

## 컴포넌트 설명

### 1. Client Side

#### 역할
- 사용자 이미지 입력 받기
- 이미지 전처리 (리사이징, 정규화)
- Head 모델로 중간 특징 추출
- 서버로 중간 특징 전송
- 결과 수신 및 표시

#### 주요 파일
- `client_inference.py`: 메인 클라이언트 로직
- `socket_client.py`: Socket 통신 모듈

### 2. Server Side

#### 역할
- 클라이언트 연결 대기
- 중간 특징 데이터 수신
- Tail 모델로 최종 추론
- 결과 판정 및 전송

#### 주요 파일
- `server_inference.py`: 메인 서버 로직

### 3. Utils

#### 모델 분할 (Model Splitter)
- `splitter.py`: InceptionResNetV2를 head와 tail로 분할
- 분할 지점: `block8_1` 레이어

### 4. Training

#### 모델 학습
- `train_model.py`: ISIC 데이터셋으로 모델 학습
- InceptionResNetV2 기반 전이 학습
- 최고 성능 모델 자동 저장

## 데이터 흐름

```
1. [Client] 이미지 입력 (150x150x3)
              ↓
2. [Client] 전처리 및 정규화
              ↓
3. [Client] Head Model 추론
              ↓ (중간 특징: 4x4x1536)
4. [Network] Socket 전송
              ↓
5. [Server] 중간 특징 수신
              ↓
6. [Server] Tail Model 추론
              ↓
7. [Server] 결과 판정 (악성/양성)
              ↓
8. [Network] 결과 메시지 전송
              ↓
9. [Client] 결과 수신 및 표시
```

## 모델 구조

### Head Model (Client)
```
Input (150, 150, 3)
    ↓
InceptionResNetV2 Layers
(input ~ block8_1)
    ↓
Output (4, 4, 1536)
```

### Tail Model (Server)
```
Input (4, 4, 1536)
    ↓
InceptionResNetV2 Layers
(block8_1 ~ end)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (128, relu, L2=0.005)
    ↓
Dropout (0.3)
    ↓
Dense (2, sigmoid)
    ↓
Output (2,) [benign_prob, malignant_prob]
```

## 통신 프로토콜

### 데이터 전송 형식

1. **데이터 크기 전송** (4 bytes)
   - 형식: unsigned int, network byte order
   - struct.pack("!I", size)

2. **실제 데이터 전송**
   - 형식: pickle 직렬화된 numpy array
   - 버퍼 크기: 4096 bytes

3. **결과 메시지 수신**
   - 형식: UTF-8 인코딩 문자열
   - 최대 크기: 1024 bytes

### 통신 시퀀스

```
Client                          Server
  │                               │
  │──── SYN ────────────────────► │
  │◄─── SYN-ACK ────────────────│
  │──── ACK ────────────────────► │
  │                               │
  │──── DATA_SIZE (4 bytes) ────► │
  │                               │
  │──── SERIALIZED_DATA ────────► │
  │                               │
  │                         Processing...
  │                               │
  │◄─── RESULT_MESSAGE ─────────│
  │                               │
  │──── FIN ────────────────────► │
  │                               │
```

## 폴더 구조 설명

```
project/
│
├── src/                    # 소스 코드
│   ├── client/             # 클라이언트 모듈
│   ├── server/             # 서버 모듈
│   ├── training/           # 학습 스크립트
│   ├── utils/              # 유틸리티 함수
│   └── config.py           # 전역 설정
│
├── models/                 # 학습된 모델 저장
│   └── *.h5               # Keras 모델 파일
│
├── data/                   # 데이터셋
│   ├── train/             # 학습 데이터
│   ├── validation/        # 검증 데이터
│   └── test/              # 테스트 데이터
│
├── assets/                 # 이미지, 시각화 자료
│   ├── *.png              # 모델 구조 이미지
│   └── *.jpg              # 테스트 이미지
│
└── docs/                   # 문서
    ├── ARCHITECTURE.md    # 이 파일
    ├── USAGE_GUIDE.md     # 사용 가이드
    └── *.pdf              # 발표 자료
```

## 성능 고려사항

### 1. 네트워크 지연
- 중간 특징 크기: 약 24KB (4x4x1536, float32)
- 전송 시간: 네트워크 속도에 따라 가변

### 2. 연산 분할
- Client: 경량 연산 (Head 모델)
- Server: 무거운 연산 (Tail 모델 + FC layers)

### 3. 확장 가능성
- 여러 클라이언트 동시 지원 가능
- 서버 멀티스레딩으로 개선 가능

## 향후 개선 방향

1. **비동기 처리**: 서버의 동시 다중 클라이언트 지원
2. **압축**: 중간 특징 데이터 압축으로 전송량 감소
3. **보안**: TLS/SSL 적용
4. **모니터링**: 실시간 성능 모니터링 대시보드
5. **최적화**: 모델 경량화 (Pruning, Quantization)

## 참고 자료

- InceptionResNetV2 논문: [Inception-v4, Inception-ResNet](https://arxiv.org/abs/1602.07261)
- ISIC Dataset: [International Skin Imaging Collaboration](https://www.isic-archive.com/)
- Split Computing: [Edge Intelligence Survey](https://arxiv.org/abs/1905.10083)
