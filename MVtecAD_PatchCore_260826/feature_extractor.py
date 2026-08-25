import torch
import torchvision.models as models

from config import BACKBONE, LAYERS


_BACKBONE_CTORS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT),
}


class ResNetFeatureExtractor:
    """
    ResNet 모델의 최종 출력(Classification)이 아닌,
    특징 추출(Feature Extraction)을 위해 중간 레이어의 출력을 가로채는 클래스입니다.

    PatchCore 알고리즘은 이미지의 공간적 구조 정보가 포함된
    중간 레이어(layer2, layer3)의 특징 맵(Feature Map)을 필요로 합니다.

    하드웨어 제약을 고려해 기본 backbone은 config.BACKBONE = "resnet18" 을 사용한다
    (원 논문은 WideResNet50을 기본으로 사용).
    """

    def __init__(self, device, backbone_name: str = BACKBONE, layers=None):
        """
        Args:
            device (torch.device): 모델을 배치할 장치 (CPU, CUDA, 또는 MPS)
            backbone_name (str): "resnet18" | "resnet34" | "resnet50" | "wide_resnet50_2"
            layers (list[str]): hook을 걸 중간 레이어 이름들. 기본값은 config.LAYERS.
        """
        if backbone_name not in _BACKBONE_CTORS:
            raise ValueError(f"지원하지 않는 backbone: {backbone_name}")

        ctor, weights = _BACKBONE_CTORS[backbone_name]
        self.model = ctor(weights=weights)

        # 중요: 모델을 평가(Evaluation) 모드로 전환합니다.
        # 이는 Dropout을 비활성화하고, BatchNorm이 학습된 통계치를 사용하도록 보장합니다.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.model.to(self.device)

        self.layers = list(layers) if layers is not None else list(LAYERS)

        # 중간 레이어의 출력을 임시로 저장할 딕셔너리
        self.features = {}

        # [핵심] Forward Hook 등록
        # ResNet은 기본적으로 마지막 레이어의 결과만 반환합니다.
        # 중간 레이어(layer2, layer3)의 출력을 가로채기 위해 'hook'을 등록합니다.
        modules = dict(self.model.named_modules())
        for name in self.layers:
            if name not in modules:
                raise ValueError(f"모델에 '{name}' 레이어가 없습니다.")
            modules[name].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        """
        클로저(Closure)를 사용하여 레이어의 이름을 기억하는 Hook 함수를 생성합니다.
        """
        def hook(module, inputs, output):
            # forward pass 중에 특정 레이어를 통과하는 시점에
            # 해당 레이어의 출력값(output)을 self.features 딕셔너리에 저장합니다.
            self.features[name] = output
        return hook

    @torch.no_grad()
    def extract(self, x):
        """
        입력 이미지로부터 지정한 레이어들의 특징 맵을 추출합니다.

        Args:
            x (torch.Tensor): [B, C, H, W] 형태의 입력 텐서

        Returns:
            dict: {layer_name: [B, C, H, W]} 형태의 딕셔너리

        Note:
            @torch.no_grad() 데코레이터는 역전파를 위한 연산 그래프를 생성하지 않아
            메모리 사용량을 줄이고 연산 속도를 높입니다(Inference 최적화).
        """
        # 매 추출 시마다 이전 결과가 남지 않도록 초기화
        self.features = {}

        # 입력 데이터를 해당 장치(GPU/MPS/CPU)로 이동
        x = x.to(self.device)

        # 모델의 Forward Pass 실행
        # 이 과정에서 등록된 hook들이 실행되어 self.features에 결과가 저장됩니다.
        self.model(x)

        # .detach()를 사용하여 연산 그래프에서 텐서를 분리합니다.
        # 이는 불필요한 메모리 누수(Memory Leak)를 방지하기 위해 필수적입니다.
        return {name: self.features[name].detach() for name in self.layers}
