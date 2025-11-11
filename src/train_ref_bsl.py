# training scripts for the ref-answer baseline model
import time 
import os
import wandb
from dataclasses import dataclass, field
from typing import List
from utils.train_utils import (
    AsagTrainer,
    AsagTrainingArguments,
    get_tokenizer
)
from utils.utils import (
    set_seed,
    eval_report,
    save_report,
)
from utils.inference import evaluate
from utils.collate import base_collate_fn
from utils.data_prep import (
    BaseLoader,
    encode_solution_pair,
    group_by_id
)
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
import torch.distributed as dist
dist.init_process_group(backend="nccl")
def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
@dataclass
class TaskArguments:
    """Task/experiment related arguments dataclass"""
    base_model: str = field(default='bert-base-multilingual-cased', metadata={"help": "base model to use"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    train_frac: float = field(default=1.0, metadata={"help": "fraction of training data to use"})
    model_class: str = field(default="ref-bsl", metadata={"help": "model class to use"})
    task_name: str = field(default="lp", metadata={"help": "name of the task"})



def main(task_args: TaskArguments, train_args: AsagTrainingArguments, custom_model_args: BackwardSupportedArguments):
    set_seed(task_args.seed)
    if not os.path.exists(train_args.save_dir):
        os.makedirs(train_args.save_dir)

    wandb.login()
    if train_args.log_wandb and is_main_process():
        wandb.init(
            config={**vars(train_args), **vars(task_args)},
            dir=train_args.save_dir,
            project="alice-benchmark",
        )
    else:
        wandb.init(mode="disabled")
    
    print(f"Training arguments: {train_args}")
    print(f"Task arguments: {task_args}")
    
    # Load the dataset
    dts_loader = BaseLoader(train_frac=task_args.train_frac, task_type=task_args.task_name)
    tokenizer = get_tokenizer(task_args.base_model)

    dts_loader.encode_all_splits(
        tokenizer=tokenizer,
        enc_fn=encode_solution_pair,
    )

    # Initialize trainer with grouped collate function
    trainer = AsagTrainer(train_args, task_args, dts_loader.train, dts_loader.val, custom_model_args=custom_model_args, multi_gpu=True)
    trainer.set_collate_fn(base_collate_fn)

    if not train_args.test_only:
        print("***** Running training *****")
        print(f"  Num examples = {len(dts_loader.train)}")
        print(f"  Num Epochs = {train_args.max_epoch}")
        print(f"  Instantaneous batch size per GPU = {train_args.batch_size}")
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test datasets
    test_model = trainer.load_model()
    inference_speed = 0
    if is_main_process():
        for test in ["test_ua", "test_uq"]:
            test_ds = getattr(dts_loader, test)
            print(f"***** Running evaluation on {test} *****")
            print(f"Num examples = {len(test_ds)}")
            
            time_start = time.time()
            
            test_predictions, test_loss = evaluate(
                test_model,
                test_ds,
                batch_size=train_args.batch_size,
                collate_fn=lambda x: trainer.collate_fn(x, pad_id=tokenizer.pad_token_id, return_meta=True)
            )
            
            inf_time = time.time() - time_start
            
            # Save predictions
            pred_dir = os.path.join(train_args.save_dir, "predictions")
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            

            test_predictions.to_csv(os.path.join(pred_dir, f"{test}_predictions.csv"), index=False)
            
            # Calculate and save metrics
            test_metrics = eval_report(test_predictions)
            save_report(test_metrics, os.path.join(pred_dir, f"{test}_metrics.json"))
            
            inference_speed += inf_time / test_predictions.shape[0]
            
            # Log metrics to wandb
            metrics_wandb = {f"{test}": test_metrics}
            wandb.log(metrics_wandb)
            
            print(f"***** {test} Results *****")
            for key, value in test_metrics.items():
                print(f"{key} = {value:.4f}")
    
    # Log final metrics
    inference_speed /= 2
    wandb.log({"inference_speed_per_sample_sec": inference_speed})
    
    print("***** Training and evaluation completed *****")

if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    main(task_args, train_args, custom_model_args)