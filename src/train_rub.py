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
    metrics_calc,
    eval_report,
    transform_for_inference,
    save_report
    )
from data_prep import Asap_Rubric,encoding_with_rubric_pair
from models import Asag_CrossEncoder, get_tokenizer


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
    # Add optimization arguments
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-epoch', default=3, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=1, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--lr2', default=3e-5, type=float, help='learning rate for the second optimizer')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--grad-accumulation-steps', default=1, type=int, help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay for Adam')
    parser.add_argument('--adam-epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
    parser.add_argument('--warmup-proportion', default=0.05, type=float, help='proportion of warmup steps')
    # Add checkpoint arguments
    parser.add_argument('--save-dir', default='results/checkpoints', help='path to save checkpoints')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to a checkpoint to load from')
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
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint, "checkpoint")
        if os.path.exists(checkpoint_path):
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.isfile(optimizer_path) and os.path.isfile(scheduler_path):
                map_location = DEFAULT_DEVICE
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
                logger.info("Loaded optimizer and scheduler from checkpoint.")

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.save_dir, "checkpoint/optimizer.pt")) and os.path.isfile(
        os.path.join(args.save_dir, "checkpoint/scheduler.pt")
    ):
        map_location = DEFAULT_DEVICE
        optimizer_path = os.path.join(args.save_dir, "checkpoint/optimizer.pt")
        scheduler_path = os.path.join(args.save_dir, "checkpoint/scheduler.pt")
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
        logger.info("Loaded the saved scheduler and optimizer.")
    return optimizer, scheduler 


    
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

def load_model(args):
    model = Asag_CrossEncoder(
        model_name=args.model_name,
        num_labels=2,
        freeze_layers=args.freeze_layers,
        freeze_embeddings=args.freeze_embeddings,
        
    )
    # if checkpoint is provided, load the model state
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint, "checkpoint", "model.pt")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            logger.info("Loaded model from checkpoint: %s", checkpoint_path)
    checkpoint_path = os.path.join(args.save_dir, "checkpoint", "model.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info("Loaded model from checkpoint: %s", checkpoint_path)
    else:
        logger.info("No checkpoint found. Initializing new model.")
    model = model.to(DEFAULT_DEVICE)
    return model
def import_cp(args, total_steps):
    # check if cp exists 
    checkpoint_path = os.path.join(args.save_dir, "checkpoint", "model.pt")
    if os.path.exists(checkpoint_path):
        logger.info("found checkpoint, loading model and optimizer")
    
    training_config_path = os.path.join(args.save_dir, "checkpoint", "training_config.json")
    if os.path.exists(training_config_path):
        with open(training_config_path, "r") as f:
            training_config = json.load(f)
        if training_config["model_name"] != args.model_name:
            logger.warning("Model type mismatch. Expected %s, but found %s", args.model_name, training_config["model_name"])
        if training_config["label_mode"] != args.label_mode:
            logger.warning("Label mode mismatch. Expected %s, but found %s", args.label_mode, training_config["label_mode"])
        
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
            collate_fn=Asap_Rubric.collate_fn,
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
        )
        eval_acc = accuracy_score(val_predictions["label_id"], val_predictions["pred_id"])
        if eval_acc > best_metric:
            best_metric = eval_acc
       
            export_cp(model, optimizer, scheduler, args, model_name="model.pt")
            logger.info("Best model saved at epoch %d", epoch + 1)
        logger.info("Epoch %d: Validation loss: %.4f, F1: %.4f, Accuracy: %.4f", epoch + 1, val_loss, eval_acc)
        wandb.log({
            "eval": {
                "loss": val_loss,
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
        collate_fn=Asap_Rubric.collate_fn,
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
        logits = model_output.logits.detach().cpu()
        prob = softmax(logits, dim=-1).numpy()
        eval_loss.append(loss.item())
        pred_id = np.argmax(logits, axis=1)
        # collect data to put in the prediction dict
        predictions["pred_id"].extend(pred_id.tolist())
        predictions["label_id"].extend(batch["label_id"].detach().cpu().numpy().tolist())
        predictions["logits"].extend(prob.tolist())
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
        break 
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
            project="sb-baseline",
            config=vars(args),
            name=f"{args.model_name}_{args.label_mode}",
            dir=args.save_dir,
        )
    else:
        wandb.init(mode="disabled")
    logger.info("Training arguments: %s", args)
    # Load the dataset
    asap = Asap_Rubric(enc_fn=encoding_with_rubric_pair)
    asap.get_encoding(tokenizer=get_tokenizer(args.model_name))
    steps_per_epoch = int(np.ceil(len(asap.train) / args.batch_size)) 
    total_steps = args.max_epoch * steps_per_epoch


 
    # Load the checkpoint 
    cp = import_cp(args, total_steps)
    model = cp["model"]
    optimizer = cp["optimizer"]
    scheduler = cp["scheduler"]

    if not args.test_only:
        model.train()
        wandb.watch(model)
        # Build optimizer and scheduler

        # Training loop
        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(asap.train))
        logger.info("  Num Epochs = %d", args.max_epoch)
        logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
        train_stats = train_epoch(
            model,
            asap.train,
            asap.val,
            optimizer,
            scheduler,
            args)  
        logger.info("***** Training finished *****")
    # Evaluate on test dataset
    



    logger.info(f"***** Running evaluation on test set *****")
    logger.info("  Num examples = %d", len(asap.test))
    test_predictions, test_loss = evaluate(
        model,
        asap.test,
        batch_size=args.batch_size,
        is_test=True,
    )
    test_predictions = transform_for_inference(test_predictions)
    test_predictions.to_csv(os.path.join(args.save_dir, "test_predictions.csv"), index=False)
    test_report = eval_report(
        test_predictions,
        group_by="EssaySet",
    )
    save_report(test_report, os.path.join(args.save_dir, "test_report.json"))
    wandb.log(test_report)
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
    