import os
import json
import logging
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix






def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(filename=None, level=logging.INFO):
    logging.basicConfig(
        filename=filename,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

def batch_to_device(batch, device):
    """
    Move the batch to the specified device.
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device) 
    return batch 


def mean_dequeue(deque):
    """
    Calculate the mean of the last N elements in a deque.
    """
    if len(deque) == 0:
        return 0
    return sum(deque) / len(deque)


def get_optimizer_step(optimizer):
    try:
        for params in optimizer.param_groups[0]["params"]:
            params_state = optimizer.state[params]
            if "step" in params_state:
                return params_state["step"]

        return -1
    except KeyError:
        return -1 
    

   
def metrics_calc(label_id, pred_id):
    """
    Calculate the metrics for the predictions.
    """
    f1 = f1_score(label_id, pred_id, average="macro")
    acc = accuracy_score(label_id, pred_id)
    metrics = {
        "f1": f1,
        "accuracy": acc,
    }
    return metrics
def metrics_calc_label(pred_id, label_id, label2id):
    """
    Calculate the F1 score for each label.
    """
    metrics = {}
    for label, label_idx in label2id.items():
        # Get binary arrays for the current label
        binary_preds = (pred_id == label_idx).astype(int)
        binary_labels = (label_id == label_idx).astype(int)
        
        # Calculate confusion matrix for the current label
        tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds, labels=[0, 1]).ravel()
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        key = "{}_f1".format(label)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[key] = f1
    overall_f1 = f1_score(label_id, pred_id, average="macro")
    overall_acc = accuracy_score(label_id, pred_id)
    metrics["overall_f1"] = overall_f1
    metrics["overall_acc"] = overall_acc
    return metrics
        
def eval_report(pred_df, label2id, group_by=None):
    """
    Report the evaluation result, print the overall F1 and accuracy to the logger.
    Additionally, create a dictionary that stores the results, sorted by the code of the datapoint,
    along with the overall metrics.
    """
    results = {}

    # Calculate overall metrics
    overall_metrics = metrics_calc_label(pred_df["pred_id"].values, pred_df["label_id"].values, label2id)
    results["overall_f1"] = overall_metrics["overall_f1"]
    results["overall_acc"] = overall_metrics["overall_acc"]

    for label, _ in label2id.items():
        results[f"overall_f1_{label}"] = overall_metrics[f"{label}_f1"]

    # Calculate metrics for each group if group_by is provided
    if group_by:
        groups = pred_df[group_by].unique()
        for group in groups:
            group_df = pred_df[pred_df[group_by] == group]
            group_preds = group_df["pred_id"].values
            group_labels = group_df["label_id"].values
            group_metrics = metrics_calc_label(group_preds, group_labels, label2id)

            results[f"{group}_f1"] = group_metrics["overall_f1"]
            results[f"{group}_acc"] = group_metrics["overall_acc"]

            for label, _ in label2id.items():
                results[f"{group}_f1_{label}"] = group_metrics[f"{label}_f1"]

    return results

def save_report(metrics, path):
    """
    Save the metrics to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4) 
def save_prediction(pred_df,id2label, path):
    """
    conver the predictions to the original labels and save them to a CSV file.
    """

    pred_df["pred_label"] = [id2label[pred] for pred in pred_df["pred_id"].values]
    with open(path, "w") as f:
        pred_df.to_csv(f, index=False) 

