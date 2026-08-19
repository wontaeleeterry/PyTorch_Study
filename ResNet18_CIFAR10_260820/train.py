import os
import time

import torch
import torch.nn as nn

from model import resnet18
from dataset import get_dataloaders

from config import DEVICE
from config import EPOCHS
from config import LEARNING_RATE
from config import MOMENTUM
from config import WEIGHT_DECAY

from config import CHECKPOINT_DIR
from config import LATEST_CHECKPOINT
from config import BEST_CHECKPOINT
from config import SAVE_INTERVAL

from utils import save_checkpoint
from utils import load_checkpoint


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += images.size(0)

        if batch_idx % 100 == 0:

            print(
                f"    "
                f"Batch [{batch_idx:4d}/"
                f"{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss / total
    )

    epoch_acc = (
        100.0 * correct / total
    )

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += images.size(0)

    loss = (
        running_loss / total
    )

    accuracy = (
        100.0 * correct / total
    )

    return loss, accuracy


def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("CIFAR-10 ResNet18 Training")
    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Epochs : {EPOCHS}"
    )

    print(
        f"Batch  : {128}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        _
    ) = get_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = resnet18(
        num_classes=10
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=LEARNING_RATE,

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 0

    best_acc = 0.0

    if os.path.exists(
        LATEST_CHECKPOINT
    ):

        print(
            "\nCheckpoint found."
        )

        print(
            "Resuming training..."
        )

        (
            start_epoch,
            best_acc
        ) = load_checkpoint(

            LATEST_CHECKPOINT,

            model,

            optimizer,

            scheduler,

            DEVICE
        )

        start_epoch += 1

        print(
            f"Resume epoch: "
            f"{start_epoch}"
        )

        print(
            f"Best accuracy: "
            f"{best_acc:.2f}%"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        start_time = time.time()

        print()
        print(
            "=" * 60
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 60
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            DEVICE
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_acc = validate(

            model,

            val_loader,

            criterion,

            DEVICE
        )

        scheduler.step()

        elapsed = (
            time.time() - start_time
        )

        print()
        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{train_acc:.2f}%"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_acc:.2f}%"
        )

        print(
            f"Time       : "
            f"{elapsed:.1f} sec"
        )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(

            model,

            optimizer,

            scheduler,

            epoch,

            best_acc,

            LATEST_CHECKPOINT
        )

        print(
            f"Latest checkpoint saved."
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_acc > best_acc:

            best_acc = val_acc

            save_checkpoint(

                model,

                optimizer,

                scheduler,

                epoch,

                best_acc,

                BEST_CHECKPOINT
            )

            print(
                f"New best model!"
            )

            print(
                f"Best Val Acc: "
                f"{best_acc:.2f}%"
            )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if (
            (epoch + 1) % SAVE_INTERVAL
            == 0
        ):

            periodic_path = (
                f"{CHECKPOINT_DIR}/"
                f"epoch_{epoch + 1:03d}.pth"
            )

            save_checkpoint(

                model,

                optimizer,

                scheduler,

                epoch,

                best_acc,

                periodic_path
            )

            print(
                f"Checkpoint saved: "
                f"{periodic_path}"
            )

    print()
    print("=" * 60)
    print("Training finished.")
    print(
        f"Best Validation Accuracy: "
        f"{best_acc:.2f}%"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()