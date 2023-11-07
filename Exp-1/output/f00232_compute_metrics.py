from typing import *
import numpy as np

from seqeval.metrics import classification_report as seqeval_report

def compute_metrics(p):
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_report(true_labels, true_predictions)
    return {
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1-score'],
        'accuracy': results['accuracy']
    }
