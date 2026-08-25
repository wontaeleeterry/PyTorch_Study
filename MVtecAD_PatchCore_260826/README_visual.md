# PatchCore for MVTec AD (ResNet18)

이 코드는 논문 *"Towards Total Recall in Industrial Anomaly Detection"* (Roth et al.,
CVPR 2022, arXiv:2106.08265)의 PatchCore 알고리즘을,
`wontaeleeterry/PyTorch_Study` 저장소의 MIMII(오디오) PatchCore 파이프라인 구조를
그대로 따라 **MVTec AD(이미지) 데이터셋**용으로 옮겨온 구현입니다.

하드웨어 부담을 줄이기 위해 원 논문 기본값인 WideResNet50 대신,
참고 저장소와 동일하게 **ResNet18** (layer2, layer3)을 기본 백본으로 사용합니다.

## 원본(MIMII) 코드와의 대응 관계

| 원본 파일 (오디오/MIMII) | 이 코드 (이미지/MVTec AD) | 비고 |
|---|---|---|
| `audio.py` (`wav_to_tensor`) | `image.py` (`image_to_tensor`, `mask_to_tensor`) | 오디오 전처리 → 이미지 전처리 |
| `dataset.py` (`split_dataset`) | `dataset.py` (`list_train_images`, `list_test_items`) | MIMII는 무작위 분할, MVTec AD는 train/test가 이미 분리되어 있어 인덱싱만 수행 |
| `feature_extractor.py` | 거의 동일 (backbone/layer를 config 기반으로 일반화) | 그대로 재사용 가능 |
| `patchcore.py` | 동일 클래스/함수명 유지, 버그 수정 + 세그멘테이션 기능 추가 | 아래 "원본 버그 수정" 참고 |
| `config.py` | MIMII 오디오 설정 → MVTec AD 이미지/카테고리 설정 | |
| `train.py` / `test.py` / `evaluate.py` | 동일한 구조(±faiss/mps 스레드 충돌 방지 코드 포함) | 카테고리 루프(`--all`) 추가, 픽셀 AUROC 평가 추가 |

### 원본 버그 수정
`patchcore.py`의 `aggregate_features()`에서
```python
layer3 = local_neighborhood_param(layer3, patch_size=patch_size)
```
로 정의되지 않은 함수(`local_neighborhood_param`)를 호출하는 오타가 있어 즉시
`NameError`가 발생했습니다. 이 코드에서는 `local_neighborhood_pooling`으로 수정하고,
`layer2`/`layer3` 두 개로 고정되어 있던 부분을 `config.LAYERS`에 지정한 임의 개수의
레이어를 사용할 수 있도록 일반화했습니다.

### MVTec AD를 위해 추가된 부분
- MIMII는 오디오 정상/이상 이진 분류만 다루지만, MVTec AD는 픽셀 단위 ground-truth
  mask가 제공되므로 `PatchCoreMemory.segmentation_map()`을 추가해 패치 점수를
  2D anomaly map으로 복원(업샘플 + Gaussian smoothing, 논문 §3.3)하도록 했습니다.
- `evaluate.py`는 이미지 레벨 AUROC/F1 뿐 아니라 픽셀 레벨 AUROC도 계산하고,
  `--all` 옵션으로 MVTec AD 15개 카테고리 평균 성능(논문 Table 1, 2 형식)을 출력합니다.

## 데이터 준비

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)를 내려받아
아래 구조로 배치합니다 (공식 배포본 구조와 동일).

```
data/mvtec_ad/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png
  cable/
    ...
```

`config.py`의 `DATA_ROOT`로 경로를 변경할 수 있습니다.

## 사용법

```bash
pip install -r requirements.txt

# 단일 카테고리
python train.py --category bottle
python test.py --category bottle
python evaluate.py --category bottle

# MVTec AD 15개 카테고리 전체 (논문 Table 1/2 형식의 평균 리포트)
python train.py --all
python test.py --all
python evaluate.py --all

# 결과 시각화 (정상/이상 판정 + 결함 위치)
python visualize.py --category bottle
python visualize.py --all
```

### 결과 시각화 (`visualize.py`)

`test.py`가 저장한 `results/<category>_test_results.csv`와
`results/<category>_segmentation.npz`를 읽어, 이미지별로 아래 내용을 시각화합니다.

- **정상/이상 판정**: `evaluate.py`와 동일한 F1-최적 threshold를 재계산해
  `score >= threshold`이면 Abnormal로 판정 (제목에 GT/Pred/OK·MISCLASSIFIED 표시).
- **결함 위치 시각화** (Abnormal로 판정된 경우):
  - anomaly score map을 원본 이미지 위에 heatmap으로 오버레이
  - Otsu 임계값(또는 `--pixel_threshold quantile`)으로 이상 영역을 이진화한 뒤
    컨투어(빨간선)와 최대 연결 영역의 바운딩박스(노란 사각형)를 표시
  - ground-truth mask가 있으면 함께 표시하고 IoU를 계산
- **요약 컨택트 시트**: 카테고리별 정상/이상 샘플을 모아 한 장으로 저장
  (`results/<category>_visualizations/_contact_sheet.png`)

출력 경로: `results/<category>_visualizations/{idx}_<defect_type>_<pred>.png`

주요 옵션:

| 옵션 | 설명 |
|---|---|
| `--category` | 단일 카테고리 지정 |
| `--all` | `config.CATEGORIES` 전체 순회 |
| `--max_images` | 개별 시각화 생성 개수 제한 (기본: 전체) |
| `--pixel_threshold {otsu,quantile}` | 결함 위치 이진화 방법 (기본 otsu) |
| `--pixel_quantile` | quantile 방식일 때 사용할 분위수 (기본 0.99) |
| `--no_contact_sheet` | 요약 시트 생성 생략 |
| `--sheet_samples` | 요약 시트에 클래스별 포함할 샘플 수 |

## 주요 하이퍼파라미터 (`config.py`)

| 항목 | 기본값 | 논문 대응 |
|---|---|---|
| `BACKBONE` | `resnet18` | 원 논문은 WideResNet50 (§4.1) — 하드웨어 고려해 경량화 |
| `LAYERS` | `["layer2", "layer3"]` | §3.1, 두 중간 계층 결합 |
| `PATCH_NEIGHBORHOOD_SIZE` | 3 | Eq.1~2, §4.4.1에서 p=3이 최적 |
| `CORESET_RATIO` | 0.10 | PatchCore-10% (§3.2, Algorithm 1) |
| `CORESET_PROJECTION_DIM` | 128 | JL 랜덤 투영 차원 (§3.2) |
| `REWEIGHT_NUM_NEIGHBORS` | 9 | Eq.7의 b (재가중치용 이웃 수) |
| `SEGMENTATION_GAUSSIAN_SIGMA` | 4.0 | §3.3, 세그멘테이션 후처리 |
| `RESIZE` / `IMAGE_SIZE` | 256 / 224 | §4.1 |

## 참고

- 논문: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022
  (arXiv:2106.08265)
- 원저자 공식 구현: https://github.com/amazon-research/patchcore-inspection
- 이 코드의 구조가 참고한 MIMII 파이프라인:
  https://github.com/wontaeleeterry/PyTorch_Study/tree/main/mimii_patchcore_greedy_coreset_selection_update_260822
