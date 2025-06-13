from sklearn.metrics import accuracy_score
import torch
import os 
import json
import argparse
import wandb 
import torch
from torch.amp import GradScaler, autocast
import logging
import numpy as np
from collections import deque, defaultdict
from models import Asag_CrossEncoder, get_tokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
from utils import (
    set_seed,
    configure_logging,
    batch_to_device,
    mean_dequeue,
    get_label_weights
    )
from data_prep_alice import Alice_Dataset
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.metrics import f1_score, accuracy_score

TASK_DATASET = Alice_Dataset
# Default device configuration
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger = logging.getLogger(__name__)
print("Using device:", DEFAULT_DEVICE)

def add_training_args(parser):
    """
    add training related args 
    """ 
    # add experiment arguments 
    parser.add_argument('--model-name', default='bert-base-uncased', type=str, help='model type to use')
    parser.add_argument('--seed', default=114514, type=int, help='random seed for initialization')
    parser.add_argument('--merge-scores', default='low', type=str, choices=['low', 'mid', 'high'], help='merge scores to binary classification')
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
    parser.add_argument('--cpt-path', default=None, type=str, help='path to a checkpoint to load from')
    # other arguments
    parser.add_argument('--dropout', type=float,default=0.1 ,metavar='D', help='dropout probability')
    parser.add_argument('--freeze-layers',default=0,type=int, metavar='F', help='number of encoder layers in bert whose parameters to be frozen')
    parser.add_argument('--freeze-embeddings', action='store_true', help='freeze the embeddings')
    parser.add_argument('--freeze-encoder', action='store_true', help='freeze the encoder')
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
    # if checkpoint path is provided, load optimizer and scheduler states
    if args.cpt_path is not None:
        checkpoint_path = os.path.join(args.cpt_path, "checkpoint")
     
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        map_location = DEFAULT_DEVICE
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
        logger.info("Loaded optimizer and scheduler from checkpoint.")

    return optimizer, scheduler 

def metrics_calc(label_id, pred_id):
    """
    Calculate the metrics for the predictions, including Quadratic Weighted Kappa (QWK), F1 score, and accuracy.
    """
    f1 = f1_score(label_id, pred_id, average='weighted')
    acc = accuracy_score(label_id, pred_id)
    
    metrics = {
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
    results["f1"] = metrics["f1"]
    results["accuracy"] = metrics["accuracy"]

    # Calculate metrics for each group if group_by is provided
    if group_by:
        grouped = pred_df.groupby(group_by)
        for group, group_df in grouped:
            group_metrics = metrics_calc(group_df["label_id"].values, group_df["pred_id"].values)
            results[f"{group}_f1"] = group_metrics["f1"]
            results[f"{group}_accuracy"] = group_metrics["accuracy"]
    return results
    
def export_cp(model, optimizer, scheduler, args, model_name="model.pt"):
 
    # save model checkpoint 
    output_dir = os.path.join(args.save_dir, "checkpoint")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    # Save a trained model
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, model_name))
    # Save training arguments
    training_config = args.__dict__.copy()
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=4) 
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

def load_model(args,label_weights=None):
    model = Asag_CrossEncoder(
        model_name=args.model_name,
        num_labels=2,
        freeze_layers=args.freeze_layers,
        freeze_embeddings=args.freeze_embeddings,
        label_weights=label_weights
        
    )
    # if checkpoint is provided, load the model state
    if args.cpt_path:
        checkpoint_path = os.path.join(args.cpt_path, "checkpoint", "model.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEFAULT_DEVICE))

    model = model.to(DEFAULT_DEVICE)
    return model
def import_cp(args, total_steps, label_weights=None):
    # check if cp exists 
    if args.cpt_path:
        logger.info("Loading checkpoint from %s", args.cpt_path)
        training_config_path = os.path.join(args.cpt_path, "checkpoint", "training_config.json")
    
        with open(training_config_path, "r") as f:
            training_config = json.load(f)
        if training_config["model_name"] != args.model_name:
            logger.warning("Model type mismatch. Expected %s, but found %s", args.model_name, training_config["model_name"])
        
    model = load_model(args,label_weights=label_weights)
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
        args): 
    model.zero_grad()
    best_metric = 0 
    loss_history = deque(maxlen=10) 
    acc_history = deque(maxlen=10)
    num_epochs = args.max_epoch + int(args.fp16 and DEFAULT_DEVICE != "cpu")
 
    train_iterator = trange(num_epochs, position=0, leave=True, desc="Epoch") 
    scaler = GradScaler(enabled=args.fp16 and DEFAULT_DEVICE == "cuda")
    for epoch in train_iterator:
        train_dataloader = DataLoader(
            train_dataset, 
            num_workers=0,
            pin_memory=True,
            batch_size=args.batch_size, 
            collate_fn=TASK_DATASET.collate_fn,
            shuffle=True) 
        epoch_iterator = tqdm(
            train_dataloader, 
            desc="Iteration", 
            position=1, 
            leave=True, 

        )

 
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
            acc = accuracy_score(label_id, pred_id)

            acc_history.append(acc)
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
        )
        eval_metrics = metrics_calc(val_predictions["label_id"], val_predictions["pred_id"])
        eval_f1 = eval_metrics["f1"]
        eval_acc = eval_metrics["accuracy"]
        if eval_f1 > best_metric:
            best_metric = eval_f1
       
            export_cp(model, optimizer, scheduler, args, model_name="model.pt")
            logger.info("Best model saved at epoch %d", epoch + 1)
        logger.info("Epoch %d: Validation loss: %.4f, F1: %.4f, Accuracy: %.4f", epoch + 1, val_loss, eval_f1, eval_acc)
        wandb.log({
            "eval": {
                "loss": val_loss,
                "f1": eval_f1,
                "accuracy": eval_acc
            }
        })
        
@torch.no_grad() 
def evaluate(
        model, 
        dataset,
        batch_size,
        is_test=False,
): 
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=TASK_DATASET.collate_fn,
        shuffle=False) 
    
    data_iterator = tqdm(dataloader, desc="Evaluating", position=0 if is_test else 2, leave=True)

 
    model.eval()
    eval_loss = []
    acc_history = deque(maxlen=10)
    predictions = defaultdict(list)
    for step, (batch, meta) in enumerate(data_iterator):
        batch = batch_to_device(batch, DEFAULT_DEVICE)
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits.detach().cpu().numpy()
        eval_loss.append(loss.item())
        pred_id = np.argmax(logits, axis=1)
        # collect data to put in the prediction dict
        predictions["pred_id"].extend(pred_id)
        predictions["label_id"].extend(batch["label_id"].detach().cpu().numpy())
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
    


def main(args):
   
    if args.freeze_encoder:
        args.freeze_layers = 114514
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    configure_logging(filename=os.path.join(args.save_dir, "train.log"))
    wandb.login()
    if args.log_wandb:
        wandb.init(
            config=vars(args),
            dir=args.save_dir,
        )
    else:
        wandb.init(mode="disabled")
    logger.info("Training arguments: %s", args)
    # Load the dataset
    ds = TASK_DATASET()
    ds.merge_scores(args.merge_scores)
    ds.get_encoding(tokenizer=get_tokenizer(args.model_name))
    steps_per_epoch = int(np.ceil(len(ds.train) / args.batch_size)) 
    total_steps = args.max_epoch * steps_per_epoch
    label_weights = get_label_weights(ds.train) if args.weighted_loss else None

 
    # Load the checkpoint 
    cp = import_cp(args, total_steps, label_weights=label_weights)
    model = cp["model"]
    optimizer = cp["optimizer"]
    scheduler = cp["scheduler"]

    if not args.test_only:
        model.train()
        wandb.watch(model)
        # Build optimizer and scheduler

        # Training loop
        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(ds.train))
        logger.info("  Num Epochs = %d", args.max_epoch)
        logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
        train_stats = train_epoch(
            model,
            ds.train,
            ds.val,
            optimizer,
            scheduler,
            args)  
        logger.info("***** Training finished *****")
    # Evaluate on test dataset
    for test in ["test_ua", "test_uq"]:
        
        test_ds = getattr(ds, test)
        logger.info(f"***** Running evaluation on {test} *****")
        logger.info("  Num examples = %d", len(test_ds))
        test_predictions, test_loss = evaluate(
            model,
            test_ds,
            batch_size=args.batch_size,
            is_test=True,
        )
        
        test_metrics = eval_report(test_predictions)
        with open(os.path.join(args.save_dir, f"{test}_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)

        metrics_wandb = {test: test_metrics}

        wandb.log(metrics_wandb)

    if args.no_save:
        logger.info("No-save flag is set. Deleting checkpoint.")
        checkpoint_dir = os.path.join(args.save_dir, "checkpoint")
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, file)
                try:
                    if file_path.endswith(".pt") and os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error("Error deleting file %s: %s", file_path, e)
     
if __name__ == "__main__":
    args = get_args()
    # Set up logging
    main(args)
    