mimii_patchcore/
│
├── data/
│   └── MIMII/
│       ├── normal/
│       │   ├── 00000000.wav
│       │   ├── 00000001.wav
│       │   └── ...
│       │
│       └── abnormal/
│           ├── 00000000.wav
│           ├── 00000001.wav
│           └── ...
│
├── memory/
├── results/
│
├── config.py
├── audio.py
├── dataset.py
├── feature_extractor.py
├── patchcore.py
├── train.py
├── test.py
├── evaluate.py
└── requirements.txt


fan/id_00/train/... 같은 MIMII 원본 디렉터리를 해석할 필요가 없습니다.
그리고 PatchCore의 학습은 정상 데이터만 사용하는 것이 중요합니다. 따라서 저는 다음과 같이 분리하는 것을 권합니다.

즉, abnormal은 train에 절대 넣지 않습니다.

전체 데이터
       │
       ├── normal
       │      ├── train  ← PatchCore Memory Bank 구축
       │      └── test   ← 정상 테스트
       │
       └── abnormal
              └── test   ← 이상 테스트

실제 train/test 파일을 물리적으로 복사해서 만들 수도 있지만, 저는 처음에는 파일을 복사하지 않고 Python에서 index를 나누는 방법을 추천합니다.

예를 들어 정상 381개라면:

normal
381개
 │
 ├── 80%
 │    304개 → train
 │
 └── 20%
      77개 → test


abnormal
381개
 │
 └── 381개 → test
 