import os
import json
import logging
import random
import numpy as np
import torch
import re 
import torch 
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

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
    

   
def metrics_calc(labels, pred_id):
    """
    Calculate the metrics for the predictions, including Quadratic Weighted Kappa (QWK), F1 score, and accuracy.
    """
    
    qwk = cohen_kappa_score(labels, pred_id, weights="quadratic")
    f1 = f1_score(labels, pred_id, average='weighted')
    acc = accuracy_score(labels, pred_id)
    conf_matrix = confusion_matrix(labels, pred_id)
    
    metrics = {
        "qwk": qwk,
        "f1": f1, 
        "accuracy": acc,
        "confusion_matrix": conf_matrix.tolist()
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
    metrics = metrics_calc(pred_df["pred_id"].values, pred_df["labels"].values)
    results["qwk"] = metrics["qwk"]
    results["f1"] = metrics["f1"]
    results["accuracy"] = metrics["accuracy"]

    # Calculate QWK for each group if group_by is provided
    if group_by:
        grouped = pred_df.groupby(group_by)
        for group, group_df in grouped:
            group_metrics = metrics_calc(group_df["labels"].values, group_df["pred_id"].values)
            results[f"{group}_qwk"] = group_metrics["qwk"]
    return results
def eval_report_gen(pred_df, group_by=None):
    def extract_ans(pred_df):
        answers = []
        for output in pred_df["pred_text"].tolist():
            output = str(output)
            match = re.search(r'\d', output)
            if match:
                answers.append(match.group())
            else:
                answers.append("NA")

        pred_df["pred_id"] = answers 
    extract_ans(pred_df)
    pred_df["labels"] = pred_df["level"].astype(str)
    answer_rate = sum(1 for ans in pred_df["pred_id"] if ans != "NA") / len(pred_df)
    results = eval_report(pred_df, group_by)
    results["answer_rate"] = answer_rate
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


def get_label_weights(dataset,label_field="labels"):
    """
    Calculate label weights for the optimizer based on the label distribution in the dataset.
    """
    label_ids = np.array(dataset[label_field])
    unique_labels, counts = torch.unique(torch.tensor(label_ids), return_counts=True)
    total_count = len(label_ids)
    w = 1 / (counts / total_count)
    return w 

def transform_for_inference(pred_df, other_filds=None):
    pred_df["logit_label"] = pred_df['logits'].apply(lambda x: float(x[1])) if len(pred_df['logits'].iloc[0]) > 1 else pred_df['logits']
    final_fields = ["id", "rubric_level", "level", "logit_label"] + (other_filds if other_filds else [])
    final_df = pred_df.loc[pred_df.groupby('id')['logit_label'].idxmax()][final_fields]
    final_df = final_df.rename(columns={'rubric_level': 'pred_id', 'level': 'labels'})
    return final_df 
def get_wandb_tag(task_args):
    ignore = ["n_labels", "use_label_weights", "seed"]
    tag = []
    for key, value in task_args.items():
        if key in ignore:
            continue
        if key == "base_model":
            tag.append(value.split("/")[-1])
        elif key == "input_fields" and value:
            tag.append("+".join(value))
        if isinstance(value, bool):
            tag.append(f"{key}={value}")
        else:
            tag.append(value)
    return tag 

if __name__ == "__main__":
    path = "results_gen/llama3.2-1b-instruct-gen/predictions/test_ua_predictions.csv"
    import pandas as pd

    pred_df = pd.read_csv(path)
    results = eval_report_gen(pred_df)
    print(results)
    