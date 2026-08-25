from pathlib import Path


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# Dataset (MVTec AD)
# ============================================================
#
# 기대하는 폴더 구조 (공식 MVTec AD 배포본과 동일):
#
#   <DATA_ROOT>/<category>/train/good/*.png
#   <DATA_ROOT>/<category>/test/good/*.png
#   <DATA_ROOT>/<category>/test/<defect_type>/*.png
#   <DATA_ROOT>/<category>/ground_truth/<defect_type>/*_mask.png

DATA_ROOT = (
    PROJECT_ROOT
    / "mvtec_ad"
)

# 단일 카테고리만 돌릴 때 사용 (train.py / test.py 기본값).
# run_all.py 는 CATEGORIES 전체를 순회한다.
CATEGORY = "bottle"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

# ============================================================
# Image preprocessing
# ============================================================
# 논문 §4.1: MVTec 이미지는 256x256으로 resize 후 224x224로 center crop.

RESIZE = 256
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42

# ============================================================
# PatchCore
# ============================================================

'''
논문의 PatchCore 구현에서 WideResNet50과 layer2, layer3 조합을 사용하는 설정이
대표적이지만, MVTec AD를 가벼운 하드웨어에서 재현할 때는 ResNet18로 먼저
pipeline을 검증하는 것이 훨씬 편하다. (MIMII 파이프라인과 동일한 선택)
'''

BACKBONE = "resnet18"
LAYERS = [
    "layer2",
    "layer3",
]

CORESET_RATIO = 0.10             # PatchCore-10% (논문 §4.2 기본 비교 설정)
CORESET_METHOD = "greedy"        # "greedy" or "random"
CORESET_PROJECTION_DIM = 128     # greedy coreset의 거리 계산용 random projection 차원
NN_K = 1

# PatchCore §3.1: locally aware patch features
# 논문 Eq(1)~(3), Figure 4 상단. p=3이 논문의 기본값이자 최적값.
PATCH_NEIGHBORHOOD_SIZE = 3

# PatchCore §3.3: anomaly score reweighting (Eq 6~7)
# 최근접 memory 벡터 m*의 "이웃 개수" b. 논문은 정확한 기본값을 명시하지
# 않아, 공식 구현(patchcore-inspection)의 기본값(b=9 근처)을 참고해 설정.
REWEIGHT_NUM_NEIGHBORS = 9

# 픽셀 레벨 세그멘테이션 후처리 (§3.3): 가우시안 스무딩 sigma
SEGMENTATION_GAUSSIAN_SIGMA = 4.0

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