import argparse
import os
import numpy as np
import wandb
from train_utils import (
    set_seed,
    configure_logging,
    import_cp,
    train_epoch,
    evaluate,
    transform_for_inference,
    eval_report,
    save_report,
    get_args
)
from data_prep_alice import (
    RubricRetrievalLoader,
    encode_with_fields,
    encode_rubric_pair,
    get_tokenizer
)

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
    collate_fn = ds.collate_fn
    tokenizer = get_tokenizer(args.base_model)
    # you can change the encoding function here
    ds.get_encoding(tokenizer=tokenizer, enc_fn=encode_rubric_pair)
    steps_per_epoch = int(np.ceil(len(ds.train) / args.batch_size)) 
    total_steps = args.max_epoch * steps_per_epoch
    # Load the checkpoint
    cp = import_cp(args, total_steps)
    model = cp["model"]
    optimizer = cp["optimizer"]
    scheduler = cp["scheduler"]
    if not args.test_only:
        model.train()
        wandb.watch(model)
        # Training loop
        print("***** Running training *****")
        print("Num examples = %d", len(ds.train))
        print("  Num Epochs = %d", args.max_epoch)
        print("  Instantaneous batch size per GPU = %d", args.batch_size)
        train_stats = train_epoch(
            model,
            ds.train,
            ds.val,
            optimizer,
            scheduler,
            args,
            collate_fn=collate_fn,
            tokenizer=tokenizer
        )
        print("***** Training finished *****")
    # Evaluate on test dataset
    for test in ["test_ua", "test_uq"]:
        
        test_ds = getattr(ds, test)
        print(f"***** Running evaluation on {test} *****")
        print("  Num examples = %d", len(test_ds))
        test_predictions, test_loss = evaluate(
            model,
            test_ds,
            batch_size=args.batch_size,
            is_test=True,
            collate_fn=collate_fn,
            tokenizer=tokenizer
        )
        test_predictions.to_csv(os.path.join(args.save_dir, f"{test}_raw_predictions.csv"), index=False)
        test_predictions = transform_for_inference(test_predictions)
        test_metrics = eval_report(test_predictions)
        save_report(test_metrics, os.path.join(args.save_dir, f"{test}_metrics.json"))
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
    