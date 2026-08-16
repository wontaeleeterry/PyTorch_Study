import torch
import torchvision.models as models


class ResNetFeatureExtractor:

    def __init__(
        self,
        device,
    ):

        weights = (
            models.ResNet18_Weights.DEFAULT
        )

        self.model = models.resnet18(
            weights=weights
        )

        self.model.eval()

        self.device = device

        self.model.to(
            self.device
        )

        self.features = {}

        self.model.layer2.register_forward_hook(
            self._make_hook("layer2")
        )

        self.model.layer3.register_forward_hook(
            self._make_hook("layer3")
        )

    def _make_hook(
        self,
        name,
    ):

        def hook(
            module,
            inputs,
            output,
        ):

            self.features[name] = output

        return hook

    @torch.no_grad()
    def extract(
        self,
        x,
    ):
        """
        Return:
            layer2: [B, C, H, W]
            layer3: [B, C, H, W]
        """

        self.features = {}

        x = x.to(
            self.device
        )

        self.model(x)

        return {
            "layer2":
                self.features["layer2"].detach(),

            "layer3":
                self.features["layer3"].detach(),
        }