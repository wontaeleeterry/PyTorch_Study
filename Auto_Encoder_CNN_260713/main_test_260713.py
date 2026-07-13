"""
저장된 MNIST_Hybrid_Model을 불러와서
임의의 MNIST 샘플 5개에 대한 예측 결과를 그래픽으로 표현하는 코드.

사용 전 준비:
- MNIST_basic_260710.py 를 먼저 실행하여 'mnist_hybrid_model.pt' 가 생성되어 있어야 함.
- 두 파일(MNIST_basic_260710.py, MNIST_predict_visualize.py)이 같은 폴더에 있어야 함
  (MNIST_Hybrid_Model 클래스를 그대로 import 해서 재사용하기 때문).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from base.base_net import BaseNet
from networks.mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Decoder

# 학습 스크립트에서 모델 클래스를 그대로 가져와 재사용
# from MNIST_basic_260710 import MNIST_Hybrid_Model

# 한번더 작성해준다. (260710)
# Hybrid model with both autoencoder and classifier capabilities
class MNIST_Hybrid_Model(BaseNet):
    def __init__(self, rep_dim=32):
        super().__init__()
        
        self.rep_dim = rep_dim
        # Autoencoder components
        self.encoder = MNIST_LeNet(rep_dim=rep_dim)
        self.decoder = MNIST_LeNet_Decoder(rep_dim=rep_dim)
        
        # Classifier component
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 10 classes for MNIST
        )
        
    def forward(self, x):
        # Get encoded features
        code = self.encoder(x)
        
        # Reconstruct original image
        reconstructed = self.decoder(code)
        
        # Classify the encoded features
        class_output = self.classifier(code)
        
        return reconstructed, class_output


def load_model(model_path='mnist_hybrid_model_260710.pt', device='cpu'):
    """저장된 체크포인트로부터 모델을 복원한다."""
    checkpoint = torch.load(model_path, map_location=device)

    model = MNIST_Hybrid_Model(rep_dim=checkpoint['rep_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"모델 로드 완료: {model_path}")
    if 'test_accuracy' in checkpoint:
        print(f"  (학습 시 테스트 정확도: {checkpoint['test_accuracy']:.2f}%)")

    return model


def get_random_samples(n_samples=5, root='./datasets'):
    """MNIST 테스트셋에서 임의의 샘플 n개를 가져온다."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)

    indices = torch.randperm(len(test_dataset))[:n_samples]
    images, labels = [], []
    for idx in indices:
        img, label = test_dataset[idx]
        images.append(img)
        labels.append(label)

    images = torch.stack(images)  # (n_samples, 1, 28, 28)
    labels = torch.tensor(labels)
    return images, labels


def unnormalize(img_tensor, mean=0.1307, std=0.3081):
    """정규화된 텐서를 다시 0~1 범위의 이미지로 되돌린다 (시각화용)."""
    return img_tensor * std + mean


def visualize_predictions(model, images, labels, device='cpu', save_path='mnist_predictions.png'):
    """실제 이미지와 모델의 예측 결과를 함께 그래픽으로 표현한다."""
    n_samples = images.size(0)
    images_device = images.to(device)

    with torch.no_grad():
        reconstructed, class_output = model(images_device)
        probs = torch.softmax(class_output, dim=1)
        confidences, predicted = torch.max(probs, dim=1)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    for i in range(n_samples):
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = confidences[i].item() * 100
        is_correct = (true_label == pred_label)

        # 위쪽 행: 원본 이미지
        img_show = unnormalize(images[i, 0]).clamp(0, 1).cpu().numpy()
        axes[0, i].imshow(img_show, cmap='gray')
        axes[0, i].set_title(f'real: {true_label}', fontsize=12)
        axes[0, i].axis('off')

        # 아래쪽 행: 예측 결과 텍스트 표시
        # 아래쪽 행: 모델이 복원한(reconstructed) 이미지
        recon_show = unnormalize(reconstructed[i, 0]).clamp(0, 1).detach().cpu().numpy()
        color = 'green' if is_correct else 'red'
        mark = '✓' if is_correct else '✗'
        axes[1, i].imshow(recon_show, cmap='gray')
        axes[1, i].set_title(
            f'predict: {pred_label} ({confidence:.1f}%) {mark}',
            fontsize=12, color=color
        )
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('origin', fontsize=12)
    axes[1, 0].set_ylabel('reconstructed', fontsize=12)

    fig.suptitle('MNIST predict (upper: origin / below: prediction)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"결과 이미지가 저장되었습니다: {save_path}")
    plt.show()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1) 저장된 모델 불러오기
    model = load_model('mnist_hybrid_model_260710.pt', device=device)

    # 2) 임의의 MNIST 샘플 5개 불러오기
    images, labels = get_random_samples(n_samples=5, root='./data')

    # 3) 예측 및 시각화
    visualize_predictions(model, images, labels, device=device)


if __name__ == '__main__':
    main()

