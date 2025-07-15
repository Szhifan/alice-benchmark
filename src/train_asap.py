import argparse
import os
import numpy as np
import wandb
from train_utils import (
    set_seed,
    configure_logging,
    import_cp,
    get_tokenizer,
    train_epoch,
    evaluate,
    transform_for_inference,
    eval_report,
    save_report,
    get_args
)
from data_prep_asap import (
    AsapRubric,

)



def main(args):
   
    if args.freeze_encoder:
        args.freeze_layers = 114514
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
    ds = AsapRubric(train_frac=args.train_frac)
    collate_fn = ds.collate_fn
    ds.get_encoding(tokenizer=get_tokenizer(args.model_name))
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
        # Build optimizer and scheduler

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
            collate_fn=collate_fn
        )
        print("***** Training finished *****")
    # Evaluate on test dataset
    print(f"***** Running evaluation on test set *****")
    print("  Num examples = %d", len(ds.test))
    test_predictions, test_loss = evaluate(
        model,
        ds.test,
        batch_size=args.batch_size,
        is_test=True,
        collate_fn=collate_fn
    )
    test_predictions = transform_for_inference(test_predictions, other_filds=["EssaySet"])
    test_predictions.to_csv(os.path.join(args.save_dir, "test_predictions.csv"), index=False)
    test_report = eval_report(
        test_predictions,
        group_by="EssaySet",
    )
    save_report(test_report, os.path.join(args.save_dir, "test_report.json"))
    metrics_wandb = {"test": test_report}
    wandb.log(metrics_wandb)
  
    if args.no_save:
        print("No-save flag is set. Deleting checkpoint.")
        checkpoint_dir = os.path.join(args.save_dir, "checkpoint")
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, file)
                try:
                    if file_path.endswith(".pt") and os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print("Error deleting file %s: %s", file_path, e)
     
if __name__ == "__main__":
    args = get_args()
    # Set up logging
    main(args)
    