import torch

from model import resnet18
from dataset import get_dataloaders

from config import DEVICE
from config import BEST_CHECKPOINT

from utils import load_checkpoint


@torch.no_grad()
def test(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

        for i in range(
            targets.size(0)
        ):

            label = targets[i].item()

            class_total[label] += 1

            if predictions[i] == targets[i]:

                class_correct[label] += 1

    accuracy = (
        100.0 * correct / total
    )

    print()
    print("=" * 60)
    print(
        f"Test Accuracy: "
        f"{accuracy:.2f}%"
    )
    print("=" * 60)

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    print()

    for i in range(10):

        class_acc = (
            100.0 *
            class_correct[i] /
            class_total[i]
        )

        print(
            f"{class_names[i]:12s} "
            f"{class_acc:6.2f}% "
            f"({class_correct[i]}/"
            f"{class_total[i]})"
        )

    return accuracy


def main():

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        _,
        _,
        test_loader
    ) = get_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = resnet18(
        num_classes=10
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint: "
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch'] + 1}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint['best_acc']:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test(
        model,
        test_loader,
        DEVICE
    )


if __name__ == "__main__":
    main()