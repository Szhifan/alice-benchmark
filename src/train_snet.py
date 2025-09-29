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
    transform_for_inference,
    
)
from inference import evaluate
from collate import snet_collate_fn
from data_prep import (
    BaseLoader,
    encode_special_tokens_snet
)
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
import torch.distributed as dist
# dist.init_process_group(backend='nccl')
def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
@dataclass
class TaskArguments:
    """Task/experiment related arguments dataclass"""
    base_model: str = field(default='bert-base-uncased', metadata={"help": "base model to use"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    n_labels: int = field(default=2, metadata={"help": "number of labels for classification"})
    train_frac: float = field(default=1.0, metadata={"help": "fraction of training data to use"})
    input_fields: List[str] = field(default_factory=lambda: ['a'])
    model_class: str = field(default='snet', metadata={"help": "model class to use"})
    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.model_class in ['xnet', 'snet', 'gen'], f"model_class must be one of ['xnet', 'snet','gen'], got {self.model_class}"
        assert self.n_labels > 0, "n_labels must be positive"
        assert 0 < self.train_frac <= 1.0, "train_frac must be between 0 and 1"
        assert all(field in ['a', 'q', 's'] for field in self.input_fields), f"input_fields must be a subset of ['a', 'q', 's'], got {self.input_fields}"
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
    print("Training arguments: %s", train_args)
    # Load the dataset
    dts_loader = BaseLoader(train_frac=task_args.train_frac)
    tokenizer = get_tokenizer(task_args.base_model)
    dts_loader.encode_all_splits(tokenizer=tokenizer, enc_fn=encode_special_tokens_snet, fields=convert_field(task_args.input_fields))
    trainer = AsagTrainer(train_args, task_args, dts_loader.train, dts_loader.val, custom_model_args=custom_model_args)

    if not train_args.test_only:
        print("***** Running training *****")
        print("Num examples = %d", len(dts_loader.train))
        print("  Num Epochs = %d", train_args.max_epoch)
        print("  Instantaneous batch size per GPU = %d", train_args.batch_size)
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test dataset
    test_model = trainer.load_model()
    inference_speed = 0
    if is_main_process():
        for test in ["test_ua", "test_uq"]:
            test_ds = getattr(dts_loader, test)
            print(f"***** Running evaluation on {test} *****")
            print("  Num examples = %d", len(test_ds))
            time_start = time.time()
            test_predictions, test_loss = evaluate(
                test_model,
                test_ds,
                batch_size=train_args.batch_size,
                collate_fn=lambda x: trainer.collate_fn(x, pad_id=tokenizer.pad_token_id, return_meta=True)
            )
            inf_time = time.time() - time_start
            pred_dir = os.path.join(train_args.save_dir, "predictions")
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            test_predictions.to_csv(os.path.join(pred_dir, f"{test}_predictions.csv"), index=False)
            test_metrics = eval_report(test_predictions)
            save_report(test_metrics, os.path.join(pred_dir, f"{test}_metrics.json"))
            inference_speed += inf_time / test_predictions.shape[0]
            metrics_wandb = {test: test_metrics}
            wandb.log(metrics_wandb)
    inference_speed /= 2
    wandb.log({"inference_speed_per_sample_sec": inference_speed})
if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    train_args.use_lora = True
    main(task_args, train_args, custom_model_args)