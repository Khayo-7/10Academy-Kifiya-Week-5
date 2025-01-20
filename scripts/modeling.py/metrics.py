from sklearn.metrics import precision_recall_fscore_support

# Define Metrics
def compute_metrics(pred, label_mapping):
    predictions, labels = pred
    predictions = predictions.argmax(axis=-1)

    true_labels = [[label for label in example if label != -100] for example in labels]
    true_predictions = [
        [label_mapping.int2str(pred) for idx, pred in enumerate(prediction) if labels[i][idx] != -100]
        for i, prediction in enumerate(predictions)
    ]
    return {
        "precision": precision_recall_fscore_support(true_labels, true_predictions, average="weighted")[:3]
    }
