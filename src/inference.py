from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import pandas as pd
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import batch_to_device, mean_dequeue
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad() 
def evaluate(model, dataset, batch_size, collate_fn=None): 
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False) 

    data_iterator = tqdm(dataloader, desc="Evaluating", position=0)

    model.to(DEFAULT_DEVICE)
    model.eval()
    eval_loss = []
    acc_history = deque(maxlen=10)
    predictions = defaultdict(list)
    for step, (batch, meta) in enumerate(data_iterator):
        batch = batch_to_device(batch, DEFAULT_DEVICE)
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits.detach().cpu()
        eval_loss.append(loss.item())
        pred_id = np.argmax(logits, axis=1)
        # collect data to put in the prediction dict
        predictions["pred_id"].extend(pred_id.tolist())
        predictions["labels"].extend(batch["labels"].detach().cpu().numpy().tolist())
        predictions["logits"].extend(logits.tolist())
        acc = accuracy_score(batch["labels"].detach().cpu().numpy(), pred_id)
        acc_history.append(acc)
        data_iterator.set_description(
            "Evaluating: loss {:.4f} acc {:.4f} ≈".format(
                mean_dequeue(eval_loss),
                mean_dequeue(acc_history),
            )
        )
        for key, value in meta.items():
            predictions[key].extend(value)
    pred_df = pd.DataFrame(predictions)
    eval_loss = np.mean(eval_loss)
    return pred_df, eval_loss 