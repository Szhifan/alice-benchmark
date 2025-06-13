import os
import json
import logging
import random
import numpy as np
import torch
from torch.optim import AdamW
import torch 
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from sklearn.metrics import cohen_kappa_score






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
    Calculate the metrics for the predictions, including Quadratic Weighted Kappa (QWK).
    """

    qwk = cohen_kappa_score(label_id, pred_id, weights="quadratic")
    metrics = {
        "qwk": qwk,
    }
    return metrics

        
def eval_report(pred_df, group_by=None):
    """
    Report the evaluation result, print the overall F1 and accuracy to the logger.
    Additionally, create a dictionary that stores the results, sorted by the code of the datapoint,
    along with the overall metrics.
    """
    results = {}

    # Calculate overall metrics
    metrics = metrics_calc(pred_df["pred_id"].values, pred_df["label_id"].values)
    results["qwk"] = metrics["qwk"]

    # Calculate QWK for each group if group_by is provided
    if group_by:
        grouped = pred_df.groupby(group_by)
        for group, group_df in grouped:
            group_metrics = metrics_calc(group_df["label_id"].values, group_df["pred_id"].values)
            results[f"{group}_qwk"] = group_metrics["qwk"]
    return results

def save_report(metrics, path):
    """
    Save the metrics to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4) 
def save_prediction(pred_df,id2label,path):
    """
    conver the predictions to the original labels and save them to a CSV file.
    """

    pred_df["pred_label"] = [id2label[pred] for pred in pred_df["pred_id"].values]
    with open(path, "w") as f:
        pred_df.to_csv(f, index=False) 


def get_label_weights(dataset,label_field="label_id"):
    """
    Calculate label weights for the optimizer based on the label distribution in the dataset.
    """
    label_ids = np.array(dataset[label_field])
    unique_labels, counts = torch.unique(torch.tensor(label_ids), return_counts=True)
    total_count = len(label_ids)
    w = 1 / (counts / total_count)
    return w 

def transform_for_inference_asap(pred_df):
    pred_df["logit_label_1"] = pred_df['logits'].apply(lambda x: float(x[1]))
    final_df = pred_df.loc[pred_df.groupby('id')['logit_label_1'].idxmax()][['id','rubric_level', 'level', 'EssaySet', 'EssayText']]
    final_df = final_df.rename(columns={'rubric_level': 'pred_id', 'level': 'label_id'})
    return final_df 

def transform_for_inference_alice(pred_df):
    pred_df["logit_label_1"] = pred_df['logits'].apply(lambda x: float(x[1]))
    final_df = pred_df.loc[pred_df.groupby('id')['logit_label_1'].idxmax()][['id', 'question_id', 'sid', 'rubric_level', 'level', "answer"]]
    final_df = final_df.rename(columns={'rubric_level': 'pred_id', 'level': 'label_id'})
    return final_df