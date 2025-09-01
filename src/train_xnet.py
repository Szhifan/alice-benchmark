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
    get_wandb_tag,
    
)
from inference import evaluate
from data_prep import (
    RubricRetrievalLoader,
    encode_fields_special_tokens,
    encode_rubric_pair,
    encode_with_fields_separate_rubric,
    encode_rubric_separate
)
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
@dataclass
class TaskArguments:
    """Task/experiment related arguments dataclass"""
    base_model: str = field(default='bert-base-uncased', metadata={"help": "base model to use"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    n_labels: int = field(default=2, metadata={"help": "number of labels for classification"})
    train_frac: float = field(default=1.0, metadata={"help": "fraction of training data to use"})
    input_fields: List[str] = field(default=None, 
                                   metadata={"help": "fields to use as input for the model"})
    model_class: str = field(default='xnet', metadata={"help": "model class to use"})
    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.model_class in ['xnet', 'snet'], f"model_class must be one of ['xnet', 'snet'], got {self.model_class}"
        assert self.n_labels > 0, "n_labels must be positive"
        assert 0 < self.train_frac <= 1.0, "train_frac must be between 0 and 1"
        assert not self.input_fields or all(field in ['a', 'r', 'q', 's'] for field in self.input_fields), "input_fields must be a subset of ['a', 'r', 'q', 's']"
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
    if train_args.log_wandb:
        wandb.init(
            config=vars(train_args) + vars(task_args),
            dir=train_args.save_dir,
            project="alice-benchmark",
            tags=get_wandb_tag(task_args)
        )
    else:
        wandb.init(mode="disabled")
    print("Training arguments: %s", train_args)
    # Load the dataset
    ds = RubricRetrievalLoader(train_frac=task_args.train_frac)
    tokenizer = get_tokenizer(task_args.base_model)

    if task_args.input_fields:
        input_fields = convert_field(task_args.input_fields)
        ds.get_encoding(
            tokenizer=tokenizer,
            enc_fn=encode_fields_special_tokens,
            fields=input_fields,
        )
    else:
        ds.get_encoding(tokenizer=tokenizer, enc_fn=encode_rubric_pair)
    trainer = AsagTrainer(train_args, task_args, ds.train, ds.val, custom_model_args=custom_model_args)

    if not train_args.test_only:
        print("***** Running training *****")
        print("Num examples = %d", len(ds.train))
        print("  Num Epochs = %d", train_args.max_epoch)
        print("  Instantaneous batch size per GPU = %d", train_args.batch_size)
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test dataset
    test_model = trainer.load_model()
    inference_speed = 0
    for test in ["test_ua", "test_uq"]:
        test_ds = getattr(ds, test)
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
        test_predictions.to_csv(os.path.join(pred_dir, f"{test}_raw_predictions.csv"), index=False)
        test_predictions = transform_for_inference(test_predictions)
        test_predictions.to_csv(os.path.join(pred_dir, f"{test}_predictions.csv"), index=False)
        test_metrics = eval_report(test_predictions)
        save_report(test_metrics, os.path.join(pred_dir, f"{test}_metrics.json"))
        inference_speed += inf_time / test_predictions.shape[0]
        metrics_wandb = {test: test_metrics}
        wandb.log(metrics_wandb)
    if train_args.no_save:
        print("No-save flag is set. Deleting checkpoint.")
        checkpoint_dir = os.path.join(train_args.save_dir, "checkpoint")
        if os.path.exists(checkpoint_dir):
            os.remove(checkpoint_dir)
    inference_speed /= 2
    wandb.log({"inference_speed_per_sample_sec": inference_speed})
if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    main(task_args, train_args, custom_model_args)