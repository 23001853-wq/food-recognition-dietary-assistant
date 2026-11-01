import csv
import os
from pathlib import Path

from food_assistant.config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "storage/temp/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from food_assistant.ml.datasets import write_json


def compute_metrics(truth, predicted, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return {
        "top1_accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "sample_count": len(truth),
    }


def export_metrics(output, truth, predicted, labels, timing=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = [x["class_key"] for x in labels]
    indices = list(range(len(keys)))
    metrics = compute_metrics(truth, predicted, len(keys))
    if timing:
        metrics.update(
            p50_ms=float(np.percentile(timing, 50)), p95_ms=float(np.percentile(timing, 95))
        )
    write_json(output / "metrics.json", metrics)
    report = classification_report(
        truth, predicted, labels=indices, target_names=keys, output_dict=True, zero_division=0
    )
    with (output / "classification_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_key", "precision", "recall", "f1-score", "support"])
        for key in keys:
            writer.writerow(
                [key, *[report[key][x] for x in ["precision", "recall", "f1-score", "support"]]]
            )
    matrix = confusion_matrix(truth, predicted, labels=indices)
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.5), max(5, len(keys) * 0.5)))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xlabel="Predicted label", ylabel="True label", xticks=indices, yticks=indices)
    ax.set_xticklabels(keys, rotation=90)
    ax.set_yticklabels(keys)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png")
    plt.close(fig)
    (output / "evaluation_summary.md").write_text(
        "# Evaluation\n\n" + "\n".join(f"- {k}: {v}" for k, v in metrics.items()), encoding="utf-8"
    )
    return metrics
