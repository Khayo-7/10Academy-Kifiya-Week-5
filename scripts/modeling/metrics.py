import os
import sys
import torch
import numpy as np
from evaluate import load
from typing import Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
# from seqeval.metrics import f1_score, precision_score, recall_score


# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("metrics")

# Define evaluation metric
metric = load("seqeval")
metric_accuracy = load("accuracy")

# Define metrics for evaluation
def compute_metrics(pred: Tuple, label_mapping: Dict, use_seqeval: bool = True) -> Dict:
    """
    Compute evaluation metrics for NER.
    """
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    
    # predictions = logits.argmax(axis=-1)
    # predictions = np.argmax(logits, axis=2)
    # predictions = torch.argmax(torch.tensor(logits), dim=-1)

    # true_labels = [[label_mapping[l] for l in label if l != -100] for label in labels]
    # true_predictions = [
    #     [label_mapping.int2str(pred) for idx, pred in enumerate(prediction) if labels[i][idx] != -100]
    #     for i, prediction in enumerate(predictions)
    # ]

    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_mapping[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    if use_seqeval:
        results = metric.compute(predictions=true_predictions, references=true_labels)
        # result = metric_accuracy.compute(predictions=predictions, references=labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average="weighted"
        )
        return {"precision": precision, "recall": recall, "f1": f1}
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }