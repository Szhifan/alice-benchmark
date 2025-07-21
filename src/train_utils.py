from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from torch.amp import GradScaler, autocast
import os 
import json
import argparse
import wandb 
import logging
import numpy as np
from collections import deque, defaultdict
import pandas as pd
from tqdm import tqdm, trange
from utils import (
    set_seed,
    configure_logging,
    batch_to_device,
    mean_dequeue,
    save_report,
    transform_for_inference
    )
from data_prep_asap import AsapRubric
from data_prep_alice import RubricRetrievalDataset, BaseDataset
from modelling import (AsagXNet,
                    AsagSNet,
                    PointerRubricModel,
                    AsagConfig,
                    get_tokenizer)
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# logger = logging.getLogger(__name__)
print("Using device:", DEFAULT_DEVICE)
MODEL_REGISTRY = {
    "asagxnet": AsagXNet,
    "asagsnet": AsagSNet,
    "pointer": PointerRubricModel,
}

def add_training_args(parser):
    """
    add training related args 
    """ 
    # add experiment arguments 
    parser.add_argument('--base-model', default='bert-base-uncased', type=str)
    parser.add_argument('--seed', default=114514, type=int)
    parser.add_argument('--n-labels', default=2, type=int)
    parser.add_argument('--train-frac', default=1.0, type=float)
    parser.add_argument('--model-type', default='asagxnet', type=str, 
                        choices=['asagxnet', 'asagsnet'],
                        help='type of model architecture to use')
    parser.add_argument('--use-lora', action='store_true', help='use LoRA for training')
    parser.add_argument('--merge-scores', action='store_true', help='merge scores only for binary classification')
    # Add optimization arguments
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-epoch', default=3, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=1, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--lr2', default=3e-5, type=float, help='learning rate for the second optimizer')
    parser.add_argument('--patience', default=3, type=int,help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--weighted-loss', action='store_true', help='use weighted loss')
    parser.add_argument('--grad-accumulation-steps', default=1, type=int, help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay for Adam')
    parser.add_argument('--adam-epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
    parser.add_argument('--warmup-proportion', default=0.05, type=float, help='proportion of warmup steps')
    # Add checkpoint arguments
    parser.add_argument('--save-dir', default='results/checkpoints', help='path to save checkpoints')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--model-path', default=None, type=str, help='path to the model checkpoint to load')
    # other arguments
    parser.add_argument('--dropout', type=float,default=0.1 ,metavar='D', help='dropout probability')
    parser.add_argument('--test-only', action='store_true', help='test model only')
    parser.add_argument('--fp16', action='store_true', help='use 16-bit float precision instead of 32-bit')
    parser.add_argument('--log-wandb',action='store_true', help='log experiment to wandb')
def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args


def build_optimizer(model, args,total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n
            ],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model. named_parameters() if "classifier" in n],
            "weight_decay": args.weight_decay,
            "lr": args.lr2,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * total_steps,
        num_training_steps=total_steps,
    )
    # if optimizers are found in save dir, load them
    if os.path.exists(os.path.join(args.save_dir, "checkpoint/optimizer.pt")) and os.path.exists(os.path.join(args.save_dir, "checkpoint/scheduler.pt")):
        checkpoint_path = os.path.join(args.save_dir, "checkpoint")
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        map_location = DEFAULT_DEVICE
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
        print("Loaded optimizer and scheduler from checkpoint.")


    return optimizer, scheduler 

   
def metrics_calc(label_id, pred_id):
    """
    Calculate the metrics for the predictions, including Quadratic Weighted Kappa (QWK), F1 score, and accuracy.
    """
    
    qwk = cohen_kappa_score(label_id, pred_id, weights="quadratic")
    f1 = f1_score(label_id, pred_id, average='weighted')
    acc = accuracy_score(label_id, pred_id)
    
    metrics = {
        "qwk": qwk,
        "f1": f1, 
        "accuracy": acc
    }
    return metrics

        
def eval_report(pred_df, group_by=None):
    """
    Report the evaluation result, print the overall metrics to the logger.
    Additionally, create a dictionary that stores the results, sorted by the code of the datapoint,
    along with the overall metrics.
    """
    results = {}

    # Calculate overall metrics
    metrics = metrics_calc(pred_df["label_id"].values, pred_df["pred_id"].values)
    results["qwk"] = metrics["qwk"]
    results["f1"] = metrics["f1"]
    results["accuracy"] = metrics["accuracy"]

    # Calculate metrics for each group if group_by is provided
    if group_by:
        grouped = pred_df.groupby(group_by)
        for group, group_df in grouped:
            group_metrics = metrics_calc(group_df["label_id"].values, group_df["pred_id"].values)
            results[f"{group}_qwk"] = group_metrics["qwk"]
            results[f"{group}_f1"] = group_metrics["f1"]
            results[f"{group}_accuracy"] = group_metrics["accuracy"]
    return results
    
def export_cp(model, optimizer, scheduler, args):
 
    # save model checkpoint 
    cp_dir = os.path.join(args.save_dir, "checkpoint")
    model_to_save = model.module if hasattr(model, "module") else model
    # Save a trained model
    model_to_save.save_pretrained(cp_dir)

    print("Saving model checkpoint to %s", cp_dir)
    torch.save(optimizer.state_dict(), os.path.join(cp_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(cp_dir, "scheduler.pt"))
    print("Saving optimizer and scheduler states to %s", cp_dir)

def load_model(args):
    if args.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type {args.model_type} is not supported. Choose from {list(MODEL_REGISTRY.keys())}.")
    # Load the model based on the specified type
    config = AsagConfig(
        base_model_name_or_path=args.base_model,
        n_labels=args.n_labels,
        use_lora=args.use_lora
    )
    model_class = MODEL_REGISTRY[args.model_type]
    model = model_class(config)
    if args.model_path:
        model.from_pretrained(args.model_path)
        

    model = model.to(DEFAULT_DEVICE)
    return model
def import_cp(args, total_steps):
    # check if cp exists 
    if os.path.exists(os.path.join(args.save_dir, "checkpoint")):
        print("Loading checkpoint from %s", args.save_dir)
        config_path = os.path.join(args.save_dir, "checkpoint", "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)
            assert config["architectures"].lower() == args.model_type, "Model architecture mismatch in checkpoint: expected {}, got {}".format(
                args.model_type, config["architectures"]
            )

    model = load_model(args)
    optimizer, scheduler = build_optimizer(model, args,total_steps) 
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
def train_epoch(
        model,
        train_dataset,
        val_dataset,
        optimizer,
        scheduler,
        args,
        collate_fn=None
        ): 
    model.zero_grad()
    best_metric = 0 
    loss_history = deque(maxlen=10) 
    acc_history = deque(maxlen=10)
    num_epochs = args.max_epoch + int(args.fp16 and DEFAULT_DEVICE != "cpu")
 
    train_iterator = trange(num_epochs, position=0, leave=True, desc="Epoch") 
    scaler = GradScaler(enabled=args.fp16 and DEFAULT_DEVICE == "cuda")
    for epoch in train_iterator:
        train_dataloader = DataLoader(train_dataset, num_workers=0, pin_memory=True, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True) 
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=1, leave=True)

 
        for step, (batch, _) in enumerate(epoch_iterator):
            model.train()
            batch = batch_to_device(batch, DEFAULT_DEVICE)
            with autocast(device_type=DEFAULT_DEVICE, enabled=args.fp16):  # mixed precision training
                model_output = model(**batch)
                loss = model_output.loss / args.grad_accumulation_steps  # normalize loss for gradient accumulation
                tr_loss = loss.item() * args.grad_accumulation_steps  # scale back for logging
                scaler.scale(loss).backward()
            label_id = batch["label_id"].detach().cpu().numpy()
            logits = model_output.logits.detach().cpu().numpy()

            pred_id = np.argmax(logits, axis=1)
            eval_acc = accuracy_score(label_id, pred_id)

            acc_history.append(eval_acc)
            loss_history.append(tr_loss)
            if (step + 1) % args.grad_accumulation_steps == 0:  # perform optimizer step after accumulation
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            epoch_iterator.set_description(
                "Epoch {}|Training: loss {:.4f} acc {:.4f} ≈".format(
                    epoch + 1,
                    mean_dequeue(loss_history),
                    mean_dequeue(acc_history),
            ))
            accuracy = np.mean(list(acc_history))
            wandb.log({
                "train": {
                    "loss:": tr_loss,
                    "accuracy": accuracy
                }
            })

                # Evaluate on validation dataset
        val_predictions, val_loss = evaluate(
            model,
            val_dataset,
            batch_size=args.batch_size,
            is_test=False,
            collate_fn=collate_fn
        )
        eval_acc = accuracy_score(val_predictions["label_id"], val_predictions["pred_id"])
        if eval_acc > best_metric:
            best_metric = eval_acc

            export_cp(model, optimizer, scheduler, args)
            print("Best model saved at epoch %d", epoch + 1)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {eval_acc:.4f}")
        wandb.log({
            "eval": {
                "loss": val_loss,
                "accuracy": eval_acc
            }
        })
        
@torch.no_grad() 
def evaluate(model, dataset, batch_size, is_test=False, collate_fn=None): 
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    data_iterator = tqdm(dataloader, desc="Evaluating", position=0 if is_test else 2, leave=True)

 
    model.eval()
    eval_loss = []
    acc_history = deque(maxlen=10)
    predictions = defaultdict(list)
    for step, (batch, meta) in enumerate(data_iterator):
        batch = batch_to_device(batch, DEFAULT_DEVICE)
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits.detach().cpu()
        eval_loss.append(loss.item())
        pred_id = np.argmax(logits, axis=1)
        # collect data to put in the prediction dict
        predictions["pred_id"].extend(pred_id.tolist())
        predictions["label_id"].extend(batch["label_id"].detach().cpu().numpy().tolist())
        predictions["logits"].extend(logits.tolist())
        acc = accuracy_score(batch["label_id"].detach().cpu().numpy(), pred_id)
        acc_history.append(acc)
        data_iterator.set_description(
            "Evaluating: loss {:.4f} acc {:.4f} ≈".format(
                mean_dequeue(eval_loss),
                mean_dequeue(acc_history),
            )
        )
        for key, value in meta.items():
            predictions[key].extend(value)
    pred_df = pd.DataFrame(predictions)
    eval_loss = np.mean(eval_loss)
    return pred_df, eval_loss 
    