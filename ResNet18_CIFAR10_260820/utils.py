import os

import torch


def calculate_accuracy(
    outputs,
    targets
):

    predictions = outputs.argmax(
        dim=1
    )

    correct = (
        predictions == targets
    ).sum().item()

    total = targets.size(0)

    return correct, total


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_acc,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_acc":
            best_acc
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    epoch = checkpoint["epoch"]

    best_acc = checkpoint["best_acc"]

    return epoch, best_acc