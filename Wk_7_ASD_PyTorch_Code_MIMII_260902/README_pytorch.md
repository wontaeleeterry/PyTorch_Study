# ASD (Self-Supervised Learning) — PyTorch 버전

Wilkinghoff, "Self-Supervised Learning for Anomalous Sound Detection" (ICASSP 2024)의
Mixup + StatEx + FeatEx + Sub-cluster AdaCos 파이프라인을 `./data/MIMII/normal`,
`./data/MIMII/abnormal` 구조의 wav 데이터에 맞춰 PyTorch로 재구현한 코드입니다.

## 폴더 구조 (필요한 형태)
```
data/
  MIMII/
    normal/
      xxx1.wav
      xxx2.wav
      ...
    abnormal/
      yyy1.wav
      ...
```
하위 폴더가 더 있어도 재귀적으로 모든 `*.wav`를 찾습니다.

## 설치
```
pip install torch soundfile librosa scikit-learn tqdm
```

## 실행
```
python train_asd_pytorch.py --data_root ./data/MIMII --duration_sec 10 --epochs 10
```

주요 옵션:
- `--sample_rate` : 리샘플링 목표 샘플레이트 (기본 16000)
- `--duration_sec` : 파형 고정 길이(초). 원본 논문은 18초(288000 샘플), MIMII는
  보통 10초 클립이므로 기본값을 10초로 두었습니다. 원본 clip 길이가 다르면
  자동으로 잘리거나 반복(tile)되어 길이가 맞춰집니다.
- `--epochs`, `--batch_size`, `--lr`, `--n_subclusters`
- `--mixup_prob`, `--statex_prob`, `--featex_prob` : 각 SSL 기법 적용 확률

## 원본 논문/코드 대비 달라진 점
1. **메타정보 없음 → 단일 클래스**: 원본은 기계타입×ID×속성 조합을 클래스로 썼지만,
   `normal`/`abnormal` 두 폴더만 있으므로 지도분류 클래스 수를 1로 두었습니다.
   이 경우 논문이 강조한 대로 Mixup/StatEx/FeatEx SSL 손실이 임베딩 정보량을
   채우는 주된 역할을 하게 됩니다 (Sub-cluster AdaCos가 단일 클래스 내에서
   16개 sub-cluster로 정상 데이터의 구조를 스스로 나눔).
2. **source/target 도메인 구분 없음**: k-means 배경 모델은 전체 정상 학습
   데이터 기준 단일 세트로 적용합니다 (원본은 source/target 두 세트를 따로 둠).
3. **정상만 학습**: 준지도(semi-supervised) 설정을 유지하여 `abnormal` 폴더는
   테스트에만 사용합니다. `normal` 폴더 중 일부(기본 20%)를 정상 테스트용으로
   떼어 둡니다.
4. **CNN 백본 축소**: 원본은 5단 SE-ResNet(2D)이지만 가독성을 위해 3단으로
   줄였습니다 (`asd_model.py`의 `SpectrogramBranch`). 성능이 부족하면
   `_ResBlock2D`를 더 쌓아 깊이를 늘릴 수 있습니다.
5. **손실 구성**: `main_statex+featex.py`의 최종 설정(`loss_weights=[1,0,1]`)을
   따라 (지도 손실) + (StatEx→FeatEx 순차 결합 SSL 손실, 9배 클래스)만
   사용했습니다. StatEx 단독 3배 클래스 헤드는 논문에서도 최종적으로 가중치 0으로
   비활성화되어 제외했습니다.

## 출력 (학습 시)
- `test_anomaly_scores.npy`, `test_labels.npy` : 테스트 샘플별 이상 점수와 정답 라벨
- `kmeans_centers.npy` : 정상 임베딩 기준 k-means 중심 (백엔드 판정용)
- `asd_config.json` : 재현에 필요한 설정(샘플레이트, 길이, seed, data_root 등)
- `asd_model_pytorch.pt` : 학습된 모델 가중치
- 콘솔에 AUC / pAUC 출력

## 테스트(평가) 스크립트: `test_asd_pytorch.py`

학습을 다시 하지 않고, 저장된 모델·k-means 중심·설정 파일만 불러와 평가합니다.

**모드 1 — 학습 때와 동일한 분할 재현 후 AUC/pAUC 재계산 (기본)**
```
python test_asd_pytorch.py
```
`asd_config.json`에 저장된 `data_root`/`seed`로 정상 hold-out + abnormal 분할을
그대로 재현해 다시 평가하고, 파일별 결과를 `test_results.csv`로 저장합니다.

**모드 2 — 라벨 없는 새 wav 폴더 채점**
```
python test_asd_pytorch.py --wav_dir ./data/MIMII/new_recordings
```
지정한 폴더의 모든 wav 파일에 대해 이상 점수를 계산하고, 학습에 쓰인 정상
데이터 분포의 90백분위(`--threshold_percentile`로 조정 가능)를 임계값으로 삼아
`normal`/`abnormal`을 판정합니다. 결과는 `test_results.csv`에 저장됩니다.

공통 옵션: `--model_path`, `--centers_path`, `--config`, `--data_root`(오버라이드),
`--output_csv`, `--batch_size`, `--device`

## 파일 구성
- `mimii_dataset.py` : wav 로딩 및 train/test 분할
- `ssl_augmentations.py` : Mixup / StatEx / FeatEx 구현
- `sub_cluster_adacos.py` : Sub-cluster AdaCos 손실
- `asd_model.py` : 두 갈래(spectrum / spectrogram+TMN) CNN 임베딩 네트워크
- `train_asd_pytorch.py` : 학습 + k-means 백엔드 + AUC 평가 메인 스크립트
