import time 
import os
import wandb
from dataclasses import dataclass, field
from typing import List
from train_utils import (
    AsagTrainer,
    AsagTrainingArguments,
    get_tokenizer
)
from utils import (
    set_seed,
    eval_report,
    save_report,
    get_wandb_tag,
    
)
from inference import evaluate
from data_prep import (
    RubricRetrievalLoader,
    encode_fields_special_tokens,
    encode_rubric_pair,
    encode_dataset,
    group_by_id,
    xnet_collate_fn,
    encode_with_fields,
)
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
import torch.distributed as dist
import pandas as pd
import numpy as np
dist.init_process_group(backend="nccl")
def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

@dataclass
class TaskArguments:
    """Task/experiment related arguments dataclass"""
    base_model: str = field(default='bert-base-uncased', metadata={"help": "base model to use"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    train_frac: float = field(default=1.0, metadata={"help": "fraction of training data to use"})
    input_fields: List[str] = field(default_factory=lambda: ['a', 'r'],
                                   metadata={"help": "fields to use as input for the model"})
    model_class: str = field(default="xnet", metadata={"help": "model class to use"})
    input_format:str = field(default="structured",metadata={"help":"type of input to use, structured or natural language"})
    add_instruction: bool = field(default=False,metadata={"help":"whether to add instruction to the LLM input"})

    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.model_class in ['xnet', 'snet'], f"model_class must be one of ['xnet', 'snet'], got {self.model_class}"
        assert 0 < self.train_frac <= 1.0, "train_frac must be between 0 and 1"
        assert all(field in ['a', 'r', 'q', 's'] for field in self.input_fields), "input_fields must be a subset of ['a', 'r', 'q', 's']"
        assert self.input_format in ['structured', 'natural_language'], "input_format must be one of ['structured', 'natural_language']"
                        
def convert_field(fields_input_list):
    map = {
        "a": "answer",
        "r": "rubric",
        "q": "question",
        "s": "sample_solution",
    }
    return [map[field] for field in fields_input_list if field in map]

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
    dts_loader = RubricRetrievalLoader(train_frac=task_args.train_frac)
    dts_loader.expand_with_rubric()
    tokenizer = get_tokenizer(task_args.base_model)

    input_fields = convert_field(task_args.input_fields)
    dts_loader.encode_all_splits(
        tokenizer=tokenizer,
        enc_fn=encode_with_fields,
        fields=input_fields,
        format=task_args.input_format,
        add_instruction=task_args.add_instruction
    )
    
    # Group datasets by ID for batch processing
    print("Grouping datasets by ID...")
    dts_loader.train = group_by_id(dts_loader.train)
    dts_loader.val = group_by_id(dts_loader.val)
    dts_loader.test_ua = group_by_id(dts_loader.test_ua)
    dts_loader.test_uq = group_by_id(dts_loader.test_uq)
    
    print(f"Grouped train dataset size: {len(dts_loader.train)}")
    print(f"Grouped val dataset size: {len(dts_loader.val)}")

    # Initialize trainer with grouped collate function
    trainer = AsagTrainer(train_args, task_args, dts_loader.train, dts_loader.val, custom_model_args=custom_model_args, multi_gpu=True)
    trainer.set_collate_fn(xnet_collate_fn)

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
                print(f"  {key} = {value:.4f}")
        
    # Clean up if no-save flag is set
    if train_args.no_save:
        print("No-save flag is set. Deleting checkpoint.")
        checkpoint_dir = os.path.join(train_args.save_dir, "checkpoint")
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
    
    # Log final metrics
    inference_speed /= 2
    wandb.log({"inference_speed_per_sample_sec": inference_speed})
    
    print("***** Training and evaluation completed *****")

if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    main(task_args, train_args, custom_model_args)