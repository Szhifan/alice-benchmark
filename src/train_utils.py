from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments
)
from modelling.modelling_snet import AsagSNet
from modelling.modelling_xnet import AsagXnet 
import torch
import os 
from dataclasses import dataclass, field
from inference import evaluate
from peft import LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM
import evaluate
import numpy as np
from accelerate import PartialState
import json
from functools import partial
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# logger = logging.getLogger(__name__)
print("Using device:", DEFAULT_DEVICE)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# encoding functions 
def get_tokenizer(base_model: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "llama" in base_model.lower():
        tokenizer.padding_side = "right"  
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    tokenizer.sep_token = tokenizer.sep_token or tokenizer.eos_token  # Ensure sep_token is set
    return tokenizer

@dataclass
class AsagTrainingArguments:
    """Training arguments dataclass"""
    batch_size: int = field(default=16, metadata={"help": "maximum number of sentences in a batch"})
    max_epoch: int = field(default=3, metadata={"help": "force stop training at specified epoch"})
    clip_norm: float = field(default=1.0, metadata={"help": "clip threshold of gradients"})
    lr: float = field(default=5e-5, metadata={"help": "learning rate"})
    patience: int = field(default=3, metadata={"help": "number of epochs without improvement on validation set before early stopping"})
    grad_accumulation_steps: int = field(default=1, metadata={"help": "number of updates steps to accumulate before performing a backward/update pass"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay for Adam"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "epsilon for Adam optimizer"})
    warmup_ratio: float = field(default=0.01, metadata={"help": "proportion of warmup steps"})
    save_dir: str = field(default="results/checkpoints", metadata={"help": "path to save checkpoints"})
    no_save: bool = field(default=False, metadata={"help": "don't save models or checkpoints"})
    cp_dir: str = field(default=None, metadata={"help": "path to the model checkpoint to load"})
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    test_only: bool = field(default=False, metadata={"help": "test model only"})
    bf16: bool = field(default=False, metadata={"help": "use 16-bit float precision instead of 32-bit"})
    log_wandb: bool = field(default=False, metadata={"help": "log experiment to wandb"})
    use_lora: bool = field(default=False, metadata={"help": "use LoRA for training"})
    use_bnb: bool = field(default=False, metadata={"help": "use 4-bit quantization for training"})
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epoch > 0, "max_epoch must be positive" 
        assert self.lr > 0, "learning rate must be positive"
        assert self.patience >= 0, "patience must be non-negative"
        assert self.grad_accumulation_steps > 0, "grad_accumulation_steps must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"



def print_trainable_parameters(model, use_4bit=False):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    
    # 添加量化状态检查
    quantized_layers = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
        # 检查是否量化
        if hasattr(param, 'quant_state'):
            quantized_layers += 1

    if use_4bit:
        trainable_params /= 2
    trainable_params_int = int(trainable_params)
    
    print(f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params_int:,d} || Trainable Parameters %: {100 * trainable_params / all_param:.2f}")
    
    if quantized_layers > 0:
        print(f"Quantized layers detected: {quantized_layers}")
class ModelLoader:
    """Model loading and initialization utilities."""

    def __init__(self, task_args, train_args, custom_model_args=None, device_map="auto"):
        self.task_args = task_args
        self.train_args = train_args
        self.custom_model_args = custom_model_args
        self.device_map = device_map
        self.lora_config = LoraConfig(
            r=self.train_args.lora_rank,
            lora_alpha=self.train_args.lora_alpha,
            lora_dropout=0.1,
            bias='none',
            target_modules="all-linear",
            task_type="SEQ_CLS" if self.task_args.model_class != "gen" else "CAUSAL_LM",
        )
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, 
            )
        self.use_custom_model = "llama" in self.task_args.base_model
    def _update_with_custom_config(self, config):
        """
        Update autoconfig with backwardsarguments 
        """

        for key, value in self.custom_model_args.to_dict().items():
            setattr(config, key, value)
        return config
    def _model_mapping(self, model_class):
        """Map model class string to actual model class."""
        mapping = {
            "snet": AsagSNet,
            # 'xnet' is the canonical name for the softmax-based Xnet implementation.
            "xnet": AsagXnet,
            "gen": AutoModelForCausalLM,
        }
        if model_class not in mapping:
            raise ValueError(f"Unsupported model class: {model_class}")
        return mapping[model_class]
    def init_model(self):
        config = AutoConfig.from_pretrained(self.task_args.base_model)
        if self.use_custom_model:
            print("Detected Llama model - preparing custom configuration...")
            config = self._update_with_custom_config(config)
        # save config for future reference
        os.makedirs(self.train_args.save_dir, exist_ok=True)

        self.train_args.use_bnb = (self.train_args.use_bnb and torch.cuda.is_available()) 
        self.train_args.use_lora = (self.train_args.use_lora and torch.cuda.is_available()) 
        # Use the custom implementations for snet and the softmax-style xnet.
        if self.task_args.model_class in ["snet", "xnet"]:
            config.pool_type = self.custom_model_args.pool_type if self.custom_model_args else "avg"
            config.base_model_name_or_path = self.task_args.base_model
            model_class = self._model_mapping(self.task_args.model_class)
            print(f"Initializing model of class {self.task_args.model_class}...")
            model = model_class(config,
                                lora_config=self.lora_config if self.train_args.use_lora else None,
                                bnb_config=self.bnb_config if self.train_args.use_bnb else None)
            device_value = self.device_map['']
            if isinstance(device_value, int):
                device = torch.device(f"cuda:{device_value}" if torch.cuda.is_available() else DEFAULT_DEVICE)
            elif isinstance(device_value, str) and device_value != 'cpu':
                device = torch.device(device_value if torch.cuda.is_available() else DEFAULT_DEVICE)
            else:
                device = torch.device(DEFAULT_DEVICE)
            model = model.to(device)
        elif self.task_args.model_class == "gen":
            print("Loading with AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(
                self.task_args.base_model,
                config=config,  # Pass custom config if it's a Llama model
                quantization_config=self.bnb_config if self.train_args.use_bnb else None,
                device_map=self.device_map,
            )
        model = self._init_peft_model(model)
        with open(os.path.join(self.train_args.save_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        return model
    def _init_peft_model(self, model):
        """Wrap the model with LoRA."""
        if not self.train_args.use_lora:
            return model
        # For our custom snet and xnet implementations use their internal init_peft if available.
        if self.task_args.model_class in ["snet", "xnet"]:
            model.init_peft()
        else:
            model = get_peft_model(model, self.lora_config)
        print_trainable_parameters(model, use_4bit=self.train_args.use_bnb)
        return model
    def _load_peft_model(self, cp_path: str):
        # Only support loading PEFT-wrapped causal LM checkpoints here (used for 'gen').
        if self.task_args.model_class == "gen":
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(cp_path) + '/',
                torch_dtype=torch.float16,
                device_map=self.device_map,
            )
        else:
            raise ValueError(f"Unsupported model class for PEFT in _load_peft_model: {self.task_args.model_class}")
        
        model = model.merge_and_unload()
        return model
    def __call__(self, cp_path: str, use_lora=False):
        """
        Load a model from a checkpoint path, with or without LoRA (PEFT).
        :param cp_path: Path to the model checkpoint.
        :param use_lora: Whether to load the model with LoRA (PEFT).
        :return: Loaded model.
        """
        cp_path = str(cp_path)
        config = AutoConfig.from_pretrained(cp_path + "/")
        bnb_config = self.bnb_config if self.train_args.use_bnb else None
        model = None

        if self.task_args.model_class in ["snet", "xnet"]:
            model_class = self._model_mapping(self.task_args.model_class)
            model = model_class.from_pretrained(
                cp_path,
                config=config,
                lora_config=self.lora_config if use_lora else None,
                bnb_config=bnb_config
            )
        elif self.task_args.model_class == "gen":
            if use_lora:
                model = self._load_peft_model(cp_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(cp_path, config=config)
        else:
            raise ValueError(f"Unknown model_class: {self.task_args.model_class}")
        model = model.to(dtype=torch.float32)
        return model

class AsagTrainer:
    """
    Trainer class for training and evaluating the AsagXNet, AsagSNet, or AsagXNetLlama models.
    """
    def __init__(self, train_args, task_args, train_dataset, validation_dataset=None, custom_model_args=None, multi_gpu=False):
        self.train_args = train_args
        self.task_args = task_args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        if multi_gpu:
            device_string = PartialState().process_index
            device_map = {'': device_string}
        else:
            device_map = {'': 0} if torch.cuda.is_available() else {'': 'cpu'}
        self.model_loader = ModelLoader(task_args, train_args, custom_model_args=custom_model_args, device_map=device_map)
        self.model = self.model_loader.init_model()
        self.tokenizer = get_tokenizer(task_args.base_model)
        self.multi_gpu = multi_gpu
        self.is_llm = "llama" in task_args.base_model
    def load_model(self):
        """Load the model for inference or evaluation."""
        if not self.train_args.cp_dir and not self.train_args.test_only:
            return self.model
        cp_path = self.train_args.cp_dir 
        return self.model_loader(cp_path, use_lora=self.train_args.use_lora)
    def set_collate_fn(self, collate_fn, fc_kwargs=None):
        """Set the data collate function."""
        collate_fn = partial(collate_fn, **(fc_kwargs or {}))
        self.collate_fn = collate_fn
        
    def train(self):
        print("Starting training...")
        if self.task_args.model_class == "gen":
            self.train_gen()
            return 
        metric = "eval_accuracy" if self.task_args.model_class in ["snet", "xnet"] else "eval_loss"
        train_args = TrainingArguments(
            # optimization parameters
            num_train_epochs=self.train_args.max_epoch,
            per_device_train_batch_size=self.train_args.batch_size,
            gradient_accumulation_steps=self.train_args.grad_accumulation_steps,
            learning_rate=self.train_args.lr,
            weight_decay=self.train_args.weight_decay,
            max_grad_norm=self.train_args.clip_norm,
            warmup_ratio=self.train_args.warmup_ratio,
            bf16=self.train_args.bf16,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit" if self.is_llm else "adamw_torch",
            remove_unused_columns=False,
            gradient_checkpointing=True if self.is_llm else False,
            gradient_checkpointing_kwargs = {"use_reentrant": False} if self.is_llm else None,
            # logging and saving parameters
            label_names=["labels"],
            greater_is_better=metric == "eval_accuracy",
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model=metric,
            logging_dir=os.path.join(self.train_args.save_dir, "logs"),
            logging_steps=10,
            save_strategy="best",
            eval_strategy="epoch",
            save_total_limit=1,
            output_dir=self.train_args.save_dir,
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=lambda batch: self.collate_fn(batch, self.tokenizer.pad_token_id),
            compute_metrics=compute_metrics,
        )
        trainer.train()

        self.model.save_pretrained(self.train_args.save_dir)

        trainer.model.save_pretrained(self.train_args.save_dir)
        self.tokenizer.save_pretrained(self.train_args.save_dir)
        return
    def train_gen(self):
        from trl import SFTTrainer, SFTConfig
        print("Starting generative model training...")
        
        sft_config = SFTConfig(
            output_dir=self.train_args.save_dir,
            num_train_epochs=self.train_args.max_epoch,
            per_device_train_batch_size=self.train_args.batch_size,
            gradient_accumulation_steps=self.train_args.grad_accumulation_steps,
            learning_rate=self.train_args.lr,
            weight_decay=self.train_args.weight_decay,
            max_grad_norm=self.train_args.clip_norm,
            warmup_ratio=self.train_args.warmup_ratio,
            bf16=self.train_args.bf16,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit" if self.is_llm else "adamw_torch",
            remove_unused_columns=False,
            gradient_checkpointing=True if self.is_llm else False,
            gradient_checkpointing_kwargs={"use_reentrant": False} if self.is_llm else None,
            logging_dir=os.path.join(self.train_args.save_dir, "logs"),
            logging_steps=10,
            save_strategy="best",
            eval_strategy="epoch",
            save_total_limit=1,
            packing=False,
        )
        
        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.model_loader.lora_config,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=self.collate_fn,
        )
        
        trainer.train()

        self.model.save_pretrained(self.train_args.save_dir)
        self.tokenizer.save_pretrained(self.train_args.save_dir)