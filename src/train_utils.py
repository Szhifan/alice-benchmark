from torch.optim import AdamW
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import os 
import argparse
from modelling.modelling_berts import (AsagXNet,
                    AsagSNet,
                    AsagConfig)
from modelling.modelling_bsl import AsagBsl
from modelling.modelling_llm import AsagXNetLlama
from data_prep import get_tokenizer, collate_fn, snet_collate_fn, xnet_collate_fn
from inference import evaluate
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# logger = logging.getLogger(__name__)
print("Using device:", DEFAULT_DEVICE)
MODEL_REGISTRY = {
    "asagxnet": AsagXNet,
    "asagsnet": AsagSNet,
    "asagxnetllama": AsagXNetLlama,
    "asagbsl": AsagBsl,
}
MODEL_TO_COLLATE_FN = {
    "asagxnet": xnet_collate_fn,
    "asagsnet": snet_collate_fn,
    "asagxnetllama": xnet_collate_fn,
    "asagbsl": collate_fn,
}
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def add_training_args(parser):
    """
    add training related args only
    """ 
    # llm-related args
    parser.add_argument('--use-lora', action='store_true', help='use LoRA for training')
    parser.add_argument('--use-bidirectional', action='store_true', help='use bidirectional attention')
    parser.add_argument('--use-latent-attention', action='store_true', help='use latent attention mechanism')
    # Add optimization arguments
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-epoch', default=3, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=1, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--patience', default=3, type=int,help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--grad-accumulation-steps', default=1, type=int, help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay for Adam')
    parser.add_argument('--adam-epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
    parser.add_argument('--warmup-ratio', default=0.01, type=float, help='proportion of warmup steps')
    # Add checkpoint arguments
    parser.add_argument('--save-dir', default='results/checkpoints', help='path to save checkpoints')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--cp-path', default=None, type=str, help='path to the model checkpoint to load')
    # other arguments
    parser.add_argument('--dropout', type=float,default=0.1 ,metavar='D', help='dropout probability')
    parser.add_argument('--test-only', action='store_true', help='test model only')
    parser.add_argument('--fp16', action='store_true', help='use 16-bit float precision instead of 32-bit')
    parser.add_argument('--log-wandb',action='store_true', help='log experiment to wandb')
def get_training_args():
    """
    Get training-related arguments only
    """
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args




def load_model(args):
    if args.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type {args.model_type} is not supported. Choose from {list(MODEL_REGISTRY.keys())}.")
    # Load the model based on the specified type
    config = AsagConfig(
        base_model_name_or_path=args.base_model,
        n_labels=args.n_labels,
        use_lora=args.use_lora,
        use_bidirectional=args.use_bidirectional,
        use_latent_attention=args.use_latent_attention,
        use_label_weights=args.use_label_weights,
    )
    model_class = MODEL_REGISTRY[args.model_type]
    model = model_class(config)
    if args.cp_path:
        print(f"Loading model from {args.cp_path}")
        model.from_pretrained(args.cp_path)
    model = model.to(DEFAULT_DEVICE)
    return model


def print_trainable_parameters(model, use_4bit=False):
    """Prints the number of trainable parameters in the model.
    :param model: PEFT model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2
    
    # 确保 trainable_params 是整数用于格式化
    trainable_params_int = int(trainable_params)
    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params_int:,d} || Trainable Parameters %: {100 * trainable_params / all_param:.2f}"
    )
class AsagTrainer:
    """
    Trainer class for training and evaluating the AsagXNet, AsagSNet, or AsagXNetLlama models.
    """
    def __init__(self, args, train_dataset,validation_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.model = load_model(args)
        self.tok = get_tokenizer(args.base_model)
        self.lora_config = LoraConfig(
            r=256,
            lora_alpha=256,
            lora_dropout=0.1,
            bias='none',
            target_modules="all-linear",
            task_type=None,
            modules_to_save=["classifier", "latent_attention"]
        )  
        if args.model_type not in MODEL_TO_COLLATE_FN:
            raise ValueError(f"No collate function found for model type {args.model_type}.")
        self.collate_fn = MODEL_TO_COLLATE_FN[args.model_type]

    def load_peft_model(self):
        """
        Load the PEFT model with LoRA configuration.
        """
        if self.args.use_lora:
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
            self.model = get_peft_model(self.model, self.lora_config)
        print_trainable_parameters(self.model, use_4bit=self.args.use_lora)
    def train(self):
        print("Starting training...")
        train_args = TrainingArguments(
            output_dir=self.args.save_dir,
            num_train_epochs=self.args.max_epoch,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accumulation_steps,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            max_grad_norm=self.args.clip_norm,
            warmup_ratio=self.args.warmup_ratio,
            logging_dir=os.path.join(self.args.save_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            fp16=self.args.fp16,
            lr_scheduler_type="cosine",
            report_to="wandb" if self.args.log_wandb else "none",
            optim="paged_adamw_32bit" if self.args.use_lora else "adamw_torch",
            remove_unused_columns=False,
            gradient_checkpointing=True if self.args.use_lora else False,
            label_names=["labels"],
            greater_is_better=True,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
        )
        trainer = SFTTrainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=lambda batch: self.collate_fn(batch, self.tok.pad_token_id),
            peft_config=self.lora_config if self.args.use_lora else None,
            compute_metrics=compute_metrics,  # 添加计算指标函数
        )
        train_res = trainer.train()
        
