import argparse
import os
import numpy as np
import wandb
from train_utils import (
    AsagTrainer,
    get_args
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
    encode_with_fields,
    encode_rubric_pair,
    get_tokenizer,
    xnet_collate_fn,
    snet_collate_fn,
    collate_fn
)
def find_best_checkpoint(save_dir):
    cp_list = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith("checkpoint")]
    if len(cp_list) == 0:
        return None
    if len(cp_list) == 1:
        return cp_list[0]
    for cp in cp_list:
        if "last" not in cp:
            return cp 
def main(args):
    if "t5" in args.base_model:
        args.model_type = "asagxnett5"
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
    print("Training arguments: %s", args)
    
    # Load the dataset
    ds = RubricRetrievalLoader(train_frac=args.train_frac) 
    tokenizer = get_tokenizer(args.base_model)
    # you can change the encoding function here
    ds.get_encoding(tokenizer=tokenizer, enc_fn=encode_rubric_pair)
    
    # Determine the correct collate function
    if "xnet" in args.model_type:
        collate_fn_to_use = xnet_collate_fn
    elif "snet" in args.model_type:
        collate_fn_to_use = snet_collate_fn
    else:
        collate_fn_to_use = collate_fn
    
    # Initialize trainer
    trainer = AsagTrainer(args, ds.train, ds.val)
    
    if not args.test_only:
        print("***** Running training *****")
        print("Num examples = %d", len(ds.train))
        print("  Num Epochs = %d", args.max_epoch)
        print("  Instantaneous batch size per GPU = %d", args.batch_size)
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test dataset
    cp_path = find_best_checkpoint(args.save_dir)
    if cp_path is None:
        print("No checkpoints found. Exiting.")
        return
    test_model = trainer.model.from_pretrained(cp_path)
    for test in ["test_ua", "test_uq"]:
        test_ds = getattr(ds, test)
        print(f"***** Running evaluation on {test} *****")
        print("  Num examples = %d", len(test_ds))
        test_predictions, test_loss = evaluate(
            test_model,
            test_ds,
            batch_size=args.batch_size,
            collate_fn=lambda x: collate_fn_to_use(x, pad_id=tokenizer.pad_token_id, return_meta=True)
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
    args = get_args()
    # Set up logging
    main(args)
    