import pandas as pd
import json
import random

from tqdm import tqdm

from datasets import Dataset
import wandb
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

wandb.init(
    project='asap-rubric',
    name='crossenc-bert-sg',
    config={
        'model_name': 'bert-base-multilingual-uncased',
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 5,
        'dataset': 'ALICE'
    })
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, losses

train = pd.read_csv('alice_data/ALICE_train_new.csv')

model = CrossEncoder("./crossenc-variant-1")

for dataset in ['UA', 'UQ']:
    df = pd.read_csv(f'alice_data/ALICE_{dataset}_new.csv')

    ref = [int(l) for l in df['level'].values]
    pred = [0] * len(ref)
    for i in tqdm(range(len(ref))):
        anchor = f"student answer: {df['answer'].values[i]} [SEP] question: {df['question'].values[i]}"
        closest = 0
        closest_sim = -9999.99
        for level in [0, 1, 2]:
            ru = f"scoring rubric: {json.loads(df['rubric'].values[i])[str(level)]} [SEP] question: {df['question'].values[i]}"

            sim = model.predict(
                [(anchor, ru)]
            )[0]
            if sim > closest_sim:
                closest = level
                closest_sim = sim
        pred[i] = closest

    print(dataset)
    print(f1_score(ref, pred, labels=[0, 1, 2], average='weighted'))
    print(cohen_kappa_score(ref, pred, labels=[0, 1, 2], weights='quadratic'))
    print(confusion_matrix(ref, pred, labels=[0, 1, 2]))
    

    dddf = {
        'pred': pred,
        'ref': ref
    }

    pd.DataFrame.from_dict(dddf).to_csv(f'{dataset}_crossenc_bert_results.csv')