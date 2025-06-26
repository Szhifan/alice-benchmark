import pandas as pd
import json
import random
from tqdm import tqdm

from datasets import Dataset

from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

from sentence_transformers.util import cos_sim

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments

model = SentenceTransformer("./sbert-variant-5")

for dataset in ['UA', 'UQ']:
    df = pd.read_csv(f'alice_data/ALICE_{dataset}_new.csv')

    qs = []
    a = []
    ref = [int(l) for l in df['level'].values]
    pred = [0] * len(ref)
    for i in tqdm(range(len(ref))):
        anchor = f"student answer: {df['answer'].values[i]} [SEP] question: {df['question'].values[i]}"
        anc = model.encode(anchor)
        closest = 0
        closest_euc = -9999.99
        for level in [0, 1, 2]:
            ru = f"scoring rubric: {json.loads(df['rubric'].values[i])[str(level)]} [SEP] question: {df['question'].values[i]}"

            sim = cos_sim(anc, model.encode(ru))[0][0]
            if sim > closest_euc:
                closest = level
                closest_euc = sim
        pred[i] = closest

    print(dataset)
    print(f1_score(ref, pred, labels=[0, 1, 2], average='weighted'))
    print(cohen_kappa_score(ref, pred, labels=[0, 1, 2], weights='quadratic'))
    print(confusion_matrix(ref, pred, labels=[0, 1, 2]))
    
    dddf = {
        'pred': pred,
        'ref': ref
    }

    pd.DataFrame.from_dict(dddf).to_csv(f'{dataset}_sbert_bert_results.csv')


    