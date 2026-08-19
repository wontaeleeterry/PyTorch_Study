import math

import torch
import matplotlib.pyplot as plt

from model import resnet18
from dataset import get_dataloaders

from config import DEVICE
from config import BEST_CHECKPOINT


# ============================================================
# CIFAR-10 class names
# ============================================================

CLASS_NAMES = [
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


# CIFAR-10
# dog = class index 5
DOG_CLASS = 5


# ============================================================
# CIFAR-10 normalization
# ============================================================

MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(3, 1, 1)

STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(3, 1, 1)


# ============================================================
# Image denormalization
# ============================================================

def denormalize(image):
    """
    Normalize된 CIFAR-10 이미지를
    원래 이미지 범위 [0, 1]로 복원한다.
    """

    image = image.cpu()

    image = image * STD + MEAN

    image = torch.clamp(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# Load model
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("Loading model")
    print("=" * 60)

    model = resnet18(
        num_classes=10
    )

    model = model.to(DEVICE)

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

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

    return model


# ============================================================
# Analyze Dog classification
# ============================================================

@torch.no_grad()
def analyze_dog(
    model,
    test_loader
):
    """
    CIFAR-10 test set 전체를 평가하면서
    dog 클래스에 대한 결과를 수집한다.

    Returns
    -------
    dog_images
        Dog 이미지 목록

    dog_predictions
        각 dog 이미지에 대한 예측 클래스

    dog_confidences
        각 dog 이미지에 대한 prediction confidence

    dog_accuracy
        Dog classification accuracy

    confusion
        10 x 10 confusion matrix
        row    = true class
        column = predicted class
    """

    dog_images = []
    dog_predictions = []
    dog_confidences = []

    correct = 0
    total = 0

    # ========================================================
    # Confusion matrix
    #
    # 행(row)    : 실제 클래스
    # 열(column) : 예측 클래스
    #
    # 예:
    #
    # confusion[dog, cat]
    #
    # = 실제 dog를 cat으로 분류한 개수
    # ========================================================

    confusion = torch.zeros(
        10,
        10,
        dtype=torch.int64
    )

    print()
    print("=" * 60)
    print("Analyzing test dataset")
    print("=" * 60)

    for batch_idx, (
        images,
        targets
    ) in enumerate(test_loader):

        images_device = images.to(
            DEVICE
        )

        targets_device = targets.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            images_device
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = (
            probabilities.max(
                dim=1
            )
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        for i in range(
            targets.size(0)
        ):

            target = targets[
                i
            ].item()

            prediction = predictions[
                i
            ].item()

            confusion[
                target,
                prediction
            ] += 1

        # ----------------------------------------------------
        # Dog samples
        # ----------------------------------------------------

        for i in range(
            targets.size(0)
        ):

            target = targets[
                i
            ].item()

            # Dog sample만 수집
            if target != DOG_CLASS:
                continue

            prediction = predictions[
                i
            ].item()

            confidence = confidences[
                i
            ].item()

            total += 1

            if prediction == DOG_CLASS:

                correct += 1

            dog_images.append(
                images[i].cpu()
            )

            dog_predictions.append(
                prediction
            )

            dog_confidences.append(
                confidence
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx + 1
        ) % 20 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(test_loader)} batches"
            )

    # ========================================================
    # Dog accuracy
    # ========================================================

    dog_accuracy = (
        100.0 * correct / total
    )

    return (
        dog_images,
        dog_predictions,
        dog_confidences,
        dog_accuracy,
        confusion
    )


# ============================================================
# Show Dog classification summary
# ============================================================

def show_dog_summary(
    dog_predictions,
    dog_confidences,
    accuracy
):

    correct_count = sum(
        prediction == DOG_CLASS
        for prediction in dog_predictions
    )

    incorrect_count = (
        len(dog_predictions)
        - correct_count
    )

    print()
    print("=" * 60)
    print("DOG CLASSIFICATION RESULT")
    print("=" * 60)

    print(
        f"Total dog images : "
        f"{len(dog_predictions)}"
    )

    print(
        f"Correct          : "
        f"{correct_count}"
    )

    print(
        f"Incorrect        : "
        f"{incorrect_count}"
    )

    print(
        f"Dog accuracy     : "
        f"{accuracy:.2f}%"
    )

    print("=" * 60)


# ============================================================
# Show Dog misclassification statistics
# ============================================================

def show_dog_confusion(
    confusion
):

    dog_row = confusion[
        DOG_CLASS
    ]

    print()
    print("=" * 60)
    print("DOG MISCLASSIFICATION ANALYSIS")
    print("=" * 60)

    print(
        f"{'Predicted Class':<15}"
        f"{'Count':>10}"
    )

    print("-" * 30)

    # --------------------------------------------------------
    # Prediction count를 큰 순서로 정렬
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        dog_row,
        descending=True
    )

    for index in sorted_indices:

        count = dog_row[
            index
        ].item()

        if count == 0:
            continue

        class_name = CLASS_NAMES[
            index.item()
        ]

        print(
            f"{class_name:<15}"
            f"{count:>10}"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # 오분류만 별도로 출력
    # --------------------------------------------------------

    print()
    print("Misclassified dog images:")

    for index in sorted_indices:

        index = index.item()

        count = dog_row[
            index
        ].item()

        if index == DOG_CLASS:
            continue

        if count == 0:
            continue

        print(
            f"  dog -> "
            f"{CLASS_NAMES[index]:<12} "
            f"{count:4d}"
        )


# ============================================================
# Show misclassified Dog images
# ============================================================

def show_misclassified_dogs(
    dog_images,
    dog_predictions,
    dog_confidences
):

    # --------------------------------------------------------
    # 잘못 분류된 이미지 index
    # --------------------------------------------------------

    wrong_indices = [

        i

        for i, prediction
        in enumerate(dog_predictions)

        if prediction != DOG_CLASS
    ]

    if len(wrong_indices) == 0:

        print()
        print(
            "No misclassified dog images."
        )

        return

    # --------------------------------------------------------
    # 최대 25개
    # --------------------------------------------------------

    max_images = min(
        len(wrong_indices),
        25
    )

    selected_indices = wrong_indices[
        :max_images
    ]

    cols = 5

    rows = math.ceil(
        max_images / cols
    )

    plt.figure(
        figsize=(15, 3 * rows)
    )

    for plot_idx, index in enumerate(
        selected_indices
    ):

        image = dog_images[
            index
        ]

        prediction = dog_predictions[
            index
        ]

        confidence = dog_confidences[
            index
        ]

        # ----------------------------------------------------
        # Denormalization
        # ----------------------------------------------------

        image = denormalize(
            image
        )

        image = image.permute(
            1,
            2,
            0
        ).numpy()

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.subplot(
            rows,
            cols,
            plot_idx + 1
        )

        plt.imshow(
            image
        )

        plt.axis("off")

        plt.title(
            f"True: dog\n"
            f"Pred: "
            f"{CLASS_NAMES[prediction]}\n"
            f"Conf: {confidence:.2f}"
        )

    plt.suptitle(
        "Misclassified CIFAR-10 Dog Images",
        fontsize=18
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# Show correctly classified Dog images
# ============================================================

def show_correct_dogs(
    dog_images,
    dog_predictions,
    dog_confidences
):

    # --------------------------------------------------------
    # Correct image index
    # --------------------------------------------------------

    correct_indices = [

        i

        for i, prediction
        in enumerate(dog_predictions)

        if prediction == DOG_CLASS
    ]

    if len(correct_indices) == 0:

        print(
            "No correctly classified dog images."
        )

        return

    # --------------------------------------------------------
    # 최대 25개
    # --------------------------------------------------------

    max_images = min(
        len(correct_indices),
        25
    )

    selected_indices = correct_indices[
        :max_images
    ]

    cols = 5

    rows = math.ceil(
        max_images / cols
    )

    plt.figure(
        figsize=(15, 3 * rows)
    )

    for plot_idx, index in enumerate(
        selected_indices
    ):

        image = dog_images[
            index
        ]

        confidence = dog_confidences[
            index
        ]

        # ----------------------------------------------------
        # Denormalization
        # ----------------------------------------------------

        image = denormalize(
            image
        )

        image = image.permute(
            1,
            2,
            0
        ).numpy()

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.subplot(
            rows,
            cols,
            plot_idx + 1
        )

        plt.imshow(
            image
        )

        plt.axis("off")

        plt.title(
            f"True: dog\n"
            f"Pred: dog\n"
            f"Conf: {confidence:.2f}"
        )

    plt.suptitle(
        "Correctly Classified CIFAR-10 Dog Images",
        fontsize=18
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# Show complete confusion matrix
# ============================================================

def show_confusion_matrix(
    confusion
):

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    matrix = confusion.numpy()

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "CIFAR-10 Confusion Matrix",
        fontsize=16
    )

    plt.colorbar()

    tick_marks = range(
        len(CLASS_NAMES)
    )

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    # --------------------------------------------------------
    # 숫자 표시
    # --------------------------------------------------------

    threshold = (
        matrix.max() / 2.0
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                )
            )

    plt.tight_layout()

    plt.show()


# ============================================================
# Print complete confusion matrix
# ============================================================

def print_confusion_matrix(
    confusion
):

    print()
    print("=" * 80)
    print("CIFAR-10 CONFUSION MATRIX")
    print("=" * 80)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        f"{'True / Pred':<15}",
        end=""
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name[:8]:>10}",
            end=""
        )

    print()

    print("-" * 115)

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:<15}",
            end=""
        )

        for j in range(10):

            value = confusion[
                i,
                j
            ].item()

            print(
                f"{value:>10}",
                end=""
            )

        print()

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("CIFAR-10 ResNet18 Test Visualization")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    (
        _,
        _,
        test_loader
    ) = get_dataloaders()

    # ========================================================
    # Model
    # ========================================================

    model = load_model()

    # ========================================================
    # Analyze
    # ========================================================

    (
        dog_images,
        dog_predictions,
        dog_confidences,
        dog_accuracy,
        confusion
    ) = analyze_dog(
        model,
        test_loader
    )

    # ========================================================
    # Dog summary
    # ========================================================

    show_dog_summary(
        dog_predictions,
        dog_confidences,
        dog_accuracy
    )

    # ========================================================
    # Dog confusion analysis
    # ========================================================

    show_dog_confusion(
        confusion
    )

    # ========================================================
    # Print complete confusion matrix
    # ========================================================

    print_confusion_matrix(
        confusion
    )

    # ========================================================
    # Show misclassified dog images
    # ========================================================

    show_misclassified_dogs(
        dog_images,
        dog_predictions,
        dog_confidences
    )

    # ========================================================
    # Show correctly classified dog images
    # ========================================================

    show_correct_dogs(
        dog_images,
        dog_predictions,
        dog_confidences
    )

    # ========================================================
    # Show complete confusion matrix
    # ========================================================

    show_confusion_matrix(
        confusion
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()