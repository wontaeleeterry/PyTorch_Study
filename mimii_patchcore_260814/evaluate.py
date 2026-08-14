import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)

from config import RESULT_ROOT


def main():

    result_path = (
        RESULT_ROOT
        / "Test_results.csv"
    )


    df = pd.read_csv(
        result_path
    )


    y_true = (
        df["label"].values
    )

    scores = (
        df["score"].values
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    auc = roc_auc_score(

        y_true,

        scores,
    )


    # ========================================================
    # Find best F1 threshold
    # ========================================================

    precision, recall, thresholds = (
        precision_recall_curve(

            y_true,

            scores,
        )
    )


    f1_values = (

        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-8
        )
    )


    best_index = (
        f1_values.argmax()
    )


    best_f1 = (
        f1_values[best_index]
    )


    if best_index < len(
        thresholds
    ):

        best_threshold = (
            thresholds[
                best_index
            ]
        )

    else:

        best_threshold = (
            scores.mean()
        )


    predictions = (
        scores >= best_threshold
    ).astype(int)


    final_f1 = f1_score(

        y_true,

        predictions,
    )


    print(
        "\n"
        "================================"
    )

    print(
        "MIMII Dataset / PatchCore"
    )

    print(
        "================================"
    )

    print(
        f"ROC-AUC       : "
        f"{auc:.4f}"
    )

    print(
        f"Best F1       : "
        f"{best_f1:.4f}"
    )

    print(
        f"Threshold     : "
        f"{best_threshold:.6f}"
    )

    print(
        f"Final F1      : "
        f"{final_f1:.4f}"
    )

    print(
        "================================"
    )


if __name__ == "__main__":

    main()