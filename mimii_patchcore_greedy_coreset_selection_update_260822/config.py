from pathlib import Path


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# Dataset
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "MIMII"
)

NORMAL_DIR = DATA_ROOT / "normal"
ABNORMAL_DIR = DATA_ROOT / "abnormal"

# ============================================================
# Train / Test split
# ============================================================

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ============================================================
# Audio
# ============================================================

SAMPLE_RATE = 16000
DURATION = 10.0
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

# ============================================================
# Spectrogram image
# ============================================================

IMAGE_SIZE = 224

# ============================================================
# PatchCore
# ============================================================

'''
논문의 PatchCore 구현에서 WideResNet50과 layer2, layer3 조합을 사용하는 설정이 대표적이지만, 
MIMII를 Mac에서 연구용으로 재현할 때는 ResNet18로 먼저 pipeline을 검증하는 것이 훨씬 편하다.
'''

BACKBONE = "resnet18" 
LAYERS = [
    "layer2",
    "layer3",
]
CORESET_RATIO = 0.01
CORESET_METHOD = "greedy"        # "greedy" or "random"
CORESET_PROJECTION_DIM = 128     # greedy coreset의 거리 계산용 random projection 차원
NN_K = 1

# PatchCore §3.1: locally aware patch features
# 논문 Eq(1)~(3), Figure 4 상단. p=3이 논문의 기본값이자 최적값.
PATCH_NEIGHBORHOOD_SIZE = 3

# PatchCore §3.3: anomaly score reweighting (Eq 6~7)
# 최근접 memory 벡터 m*의 "이웃 개수" b. 논문은 정확한 기본값을 명시하지
# 않아, 공식 구현(patchcore-inspection)의 기본값을 참고해 설정.
REWEIGHT_NUM_NEIGHBORS = 3

# ============================================================
# Output
# ============================================================

MEMORY_ROOT = (
    PROJECT_ROOT
    / "memory"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
)