import argparse
import os
import numpy as np
import wandb
from train_utils import (
    AsagTrainer,
    get_training_args,
    add_training_args
)
from utils import (
    set_seed,
    configure_logging,
    eval_report,
    save_report,
    transform_for_inference
)
from inference import evaluate
from data_prep import (
    RubricRetrievalLoader,
    encode_fields_special_tokens,
    encode_rubric_pair,
    get_tokenizer
)

def add_experiment_args(parser):
    """
    add experiment related args 
    """ 
    parser.add_argument('--base-model', default='bert-base-uncased', type=str)
    parser.add_argument('--seed', default=114514, type=int)
    parser.add_argument('--n-labels', default=2, type=int)
    parser.add_argument('--train-frac', default=1.0, type=float)
    parser.add_argument('--model-type', default='asagxnet', type=str, 
                        choices=['asagxnet', 'asagsnet'],
                        help='type of model architecture to use')
    parser.add_argument('--use-label-weights', action='store_true', help='use label weights for imbalanced dataset')
    parser.add_argument('--input-fields', nargs='+', default=['answer', 'rubric'], 
                        help='fields to use as input for the model, e.g. "answer rubric question"')

def get_args():
    """
    Get combined experiment and training arguments
    """
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    add_training_args(parser)
    args = parser.parse_args()
    return args

def main(args):
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    wandb.login()
    if args.log_wandb:
        wandb.init(
            config=vars(args),
            dir=args.save_dir,
        )
    else:
        wandb.init(mode="disabled")
    print("Training arguments: %s", args)
    
    # Load the dataset
    ds = RubricRetrievalLoader(train_frac=args.train_frac) 
    tokenizer = get_tokenizer(args.base_model)
    
    
    if args.input_fields:
        ds.get_encoding(
            tokenizer=tokenizer,
            enc_fn=encode_fields_special_tokens,
            fields=args.input_fields,
        )
    else:
        ds.get_encoding(tokenizer=tokenizer, enc_fn=encode_rubric_pair)


    trainer = AsagTrainer(args, ds.train, ds.val)
    
    if not args.test_only:
        print("***** Running training *****")
        print("Num examples = %d", len(ds.train))
        print("  Num Epochs = %d", args.max_epoch)
        print("  Instantaneous batch size per GPU = %d", args.batch_size)
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test dataset
    test_model = trainer.model
    for test in ["test_ua", "test_uq"]:
        test_ds = getattr(ds, test)
        print(f"***** Running evaluation on {test} *****")
        print("  Num examples = %d", len(test_ds))
        test_predictions, test_loss = evaluate(
            test_model,
            test_ds,
            batch_size=args.batch_size,
            collate_fn=lambda x: trainer.collate_fn(x, pad_id=tokenizer.pad_token_id, return_meta=True)
        )
        pred_dir = os.path.join(args.save_dir, "predictions")
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        test_predictions.to_csv(os.path.join(pred_dir, f"{test}_raw_predictions.csv"), index=False)
        test_predictions = transform_for_inference(test_predictions)
        test_predictions.to_csv(os.path.join(pred_dir, f"{test}_predictions.csv"), index=False)
        test_metrics = eval_report(test_predictions)
        save_report(test_metrics, os.path.join(pred_dir, f"{test}_metrics.json"))
        metrics_wandb = {test: test_metrics}
        wandb.log(metrics_wandb)
    if args.no_save:
        print("No-save flag is set. Deleting checkpoint.")
        checkpoint_dir = os.path.join(args.save_dir, "checkpoint")
        if os.path.exists(checkpoint_dir):
            os.remove(checkpoint_dir)
     
if __name__ == "__main__":
    experiment_args = get_args()
    main(experiment_args)