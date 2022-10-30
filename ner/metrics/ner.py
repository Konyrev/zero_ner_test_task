import evaluate
import numpy as np

from sklearn.metrics import classification_report
from torch import max, no_grad, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, Dict, List

SEQEVAL_METRIC = evaluate.load("seqeval")


def calculate_span_metrics(
        gold_labels_per_sentence: List[List[int]],
        predict_labels_per_sentence: List[List[int]],
        id_to_tag: Dict[int, str]
):
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[(id_to_tag[l] if id_to_tag else l) for l in label if l != -100] for label in gold_labels_per_sentence]
    true_predictions = [
        [(id_to_tag[p] if id_to_tag else p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predict_labels_per_sentence, gold_labels_per_sentence)
    ]
    all_metrics = SEQEVAL_METRIC.compute(predictions=true_predictions, references=true_labels)
    print('Spans')
    metrics = {
        "spans precision": all_metrics["overall_precision"],
        "spans recall": all_metrics["overall_recall"],
        "spans f1": all_metrics["overall_f1"],
        "spans accuracy": all_metrics["overall_accuracy"],
    }
    print(metrics)
    return metrics


def calculate_metrics(
        gold_labels_per_sentence: List[List[int]],
        predict_labels_per_sentence: List[List[int]],
        id_to_tag: Dict[int, str]=None
):
    if id_to_tag:
        target_names = [id_to_tag[i] for i in range(len(id_to_tag))]

    true_labels = [[(id_to_tag[l] if id_to_tag else l) for l in label if l != -100] for label in gold_labels_per_sentence]
    true_predictions = [
        [(id_to_tag[p] if id_to_tag else p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predict_labels_per_sentence, gold_labels_per_sentence)
    ]

    true_labels = [i for labels_ in true_labels for i in labels_]
    true_predictions = [i for labels_ in true_predictions for i in labels_]

    print('Tokens')
    print(classification_report(
        true_labels,
        true_predictions,
        target_names=target_names if id_to_tag else None
    ))

    metrics = classification_report(
        true_labels,
        true_predictions,
        target_names=target_names if id_to_tag else None,
        output_dict=True
    )
    result = dict()
    for label_name, label_metrics in metrics.items():
        if isinstance(label_metrics, float):
            continue

        for name, value in label_metrics.items():
            result[f'token {label_name} {name}'] = value

    return result

def transform_logits(predictions: Tensor) -> List[List[int]]:
    score, tag_seq = max(predictions, -1)
    return [label.item() for label in list(tag_seq.cpu().data)]


def transform_target(target_labels: Tensor) -> List[List[int]]:
    return [label.item() for label in list(target_labels.cpu().data)]


def check_metrics(
        dataloader: DataLoader,
        model: Module,
        id_to_tag: Dict[int, str],
        model_input_formatter: Callable = lambda input: input,
        model_output_formatter: Callable = lambda output: output,
        transform_target: Callable = transform_target,
        transform_logits: Callable = transform_logits,
):
    model.train(False)
    losses = []
    with no_grad():
        predict_labels, gold_labels = [], []
        for data in dataloader:
            loss, logits = model_output_formatter(model.forward(model_input_formatter(data)))
            gold_labels.append(transform_target(data[-1]))
            predict_labels.append(transform_logits(logits))
            losses.append(loss.item() / len(data[0]))
        calculate_metrics(gold_labels, predict_labels, id_to_tag)
        calculate_span_metrics(gold_labels, predict_labels, id_to_tag)
    return losses


def compute_metrics(eval_preds, id_to_tag: Dict[int, str]):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    result = calculate_span_metrics(labels, predictions, id_to_tag)
    result.update(calculate_metrics(labels, predictions, id_to_tag))
    return result