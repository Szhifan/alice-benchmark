import time 
import os
import wandb
from dataclasses import dataclass, field
from trainer import (
    AsagTrainer,
    AsagTrainingArguments,
    get_tokenizer
)
from utils import (
    set_seed,
    eval_report,
    save_report,  
    transform_for_inference,
    evaluate,
)
from data_utils.data_prep import xnet_collate_fn, base_collate_fn, encode_sequence_bert, encode_sequence_llm, encode_with_spans_bert, encode_with_spans_llm, grasp_collate_fn
from data_utils.data_loader_asap import (
    Asap_Loader,
    group_by_id,
)
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
import torch.distributed as dist
USE_MULTI_GPU = True
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
    model_class: str = field(default="xnet", metadata={"help": "model class to use"})
    input_format:str = field(default="structured",metadata={"help":"type of input to use, structured or natural language"})
    add_instruction: bool = field(default=False,metadata={"help":"whether to add instruction to the LLM input"})
    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.model_class in ['xnet', 'snet', 'grasp', 'tolegra', "xnet-contrastive"], f"model_class must be one of ['xnet', 'snet', 'grasp', 'tolegra', 'xnet-contrastive'], got {self.model_class}"
        assert 0 < self.train_frac <= 1.0, "train_frac must be between 0 and 1"
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

    
    if train_args.log_wandb and is_main_process():
        wandb.login()
        wandb.init(
            config={**vars(train_args), **vars(task_args)},
            dir=train_args.save_dir,
            project="asap-rubrics",
        )
    else:
        wandb.init(mode="disabled")
    
    print(f"Training arguments: {train_args}")
    print(f"Task arguments: {task_args}")
    
    # Load the dataset
    dts_loader = Asap_Loader(train_frac=task_args.train_frac)
    if task_args.model_class != "grasp":
        dts_loader.expand_with_rubric()
    tokenizer = get_tokenizer(task_args.base_model)
    is_llm = "llama" in task_args.base_model.lower() or "mistral" in task_args.base_model.lower() or "gpt" in task_args.base_model.lower()
    
    if task_args.model_class in ["grasp", "tolegra"]:
        if is_llm:
            encode_fn = lambda example: encode_with_spans_llm(
                example,
                tokenizer,
                input_fields=["answer", "rubric"],
                add_instruction=task_args.add_instruction,
                format=task_args.input_format
            )
        else:
            encode_fn = lambda example: encode_with_spans_bert(
                example,
                tokenizer,
                input_fields=["answer", "rubric"]
            )
    elif is_llm:
        encode_fn = lambda example: encode_sequence_llm(
            example,
            tokenizer,
            fields=["answer", "rubric"],
            add_instruction=task_args.add_instruction,
            format=task_args.input_format
        )
    else:
        encode_fn = lambda example: encode_sequence_bert(
            example,
            tokenizer,
            fields=["answer", "rubric"],
        )
    dts_loader.train = dts_loader.train.map(lambda x: encode_fn(x)) 
    dts_loader.val = dts_loader.val.map(lambda x: encode_fn(x))

    # Group datasets by ID for batch processing
    print("Grouping datasets by ID...")
    if task_args.model_class == "xnet":
        dts_loader.train = group_by_id(dts_loader.train)
        dts_loader.val = group_by_id(dts_loader.val)
    
    print(f"Grouped train dataset size: {len(dts_loader.train)}")
    print(f"Grouped val dataset size: {len(dts_loader.val)}")

    # Initialize trainer with appropriate collate function
    if task_args.model_class in ["grasp", "tolegra"]:
        from data_utils.data_prep import grasp_collate_fn
        collate_fn = grasp_collate_fn
    elif task_args.model_class == "xnet-contrastive":
        collate_fn = base_collate_fn
    else:
        collate_fn = xnet_collate_fn
        
    trainer = AsagTrainer(train_args, task_args, dts_loader.train, dts_loader.val, custom_model_args=custom_model_args, multi_gpu=USE_MULTI_GPU)
    trainer.set_collate_fn(collate_fn)

    if not train_args.test_only:
        print("***** Running training *****")
        print(f"  Num examples = {len(dts_loader.train)}")
        print(f"  Num Epochs = {train_args.max_epoch}")
        print(f"  Instantaneous batch size per GPU = {train_args.batch_size}")
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test datasets
    if not is_main_process():
        return
    test_model = trainer.load_model()
    inference_speed = 0
    dts_loader.test = dts_loader.test.map(lambda x: encode_fn(x))
    if task_args.model_class == "xnet":
        dts_loader.test = group_by_id(dts_loader.test)

    test_ds = dts_loader.test
    print(f"***** Running evaluation on test set *****")
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
    
    if task_args.model_class == "xnet-contrastive":
        test_predictions = transform_for_inference(test_predictions)
    test_predictions.to_csv(os.path.join(pred_dir, "test_predictions.csv"), index=False)
    
    # Calculate and save metrics
    test_metrics = eval_report(test_predictions)
    save_report(test_metrics, os.path.join(pred_dir, "test_metrics.json"))

    inference_speed += inf_time / test_predictions.shape[0]
    
    # Log metrics to wandb
    metrics_wandb = {"test": test_metrics}
    wandb.log(metrics_wandb)
    
    print("***** test Results *****")
    for key, value in test_metrics.items():
        print(f"{key} = {value:.4f}")

    # Log final metrics
    wandb.log({"inference_speed_per_sample_sec": inference_speed})
    
    print("***** Training and evaluation completed *****")

if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    main(task_args, train_args, custom_model_args)