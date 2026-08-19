cifar10_resnet18/
│
├── .venv/
│
├── data/
│   └── CIFAR-10/
│
├── checkpoints/
│   ├── latest.pth
│   ├── best.pth
│   ├── epoch_005.pth
│   ├── epoch_010.pth
│   ├── epoch_015.pth
│   └── ...
│
├── model.py
├── dataset.py
├── config.py
├── utils.py
├── train.py
├── test.py
│
└── requirements.txt


주요 기능은 다음과 같습니다.

ResNet18
CIFAR-10
CIFAR-10에 맞게 ResNet18의 첫 Conv를 3x3, stride=1로 수정
첫 MaxPool 제거
MPS 사용
학습 중 checkpoint 자동 저장
latest.pth 저장
가장 좋은 validation accuracy의 best.pth 저장
중단된 학습을 이어서 학습 가능
Train / Validation / Test 분리
학습률 scheduler
학습 결과 출력
Mac MPS에서 AMP는 일단 사용하지 않는 안정적인 구성