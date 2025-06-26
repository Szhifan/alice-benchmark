import pandas as pd
import json
import random
import wandb
from datasets import Dataset
import os
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from sentence_transformers.util import dot_score
train_config = {
    'model_name': 'bert-base-multilingual-uncased',
    'learning_rate': 1e-5,
    'batch_size': 16,
    'epochs': 5,
    'dataset': 'ALICE'} 
wandb.login(key=os.getenv('WANDB_API_KEY'))
wandb.init(
    project='alice-rubric',
    name='sbert-sg',
    config=train_config)
train = pd.read_csv('alice_data/ALICE_train_new.csv')

# Split train data into train and validation (80% train, 20% validation)
indices = list(range(len(train)))
random.shuffle(indices)
train_size = int(0.8 * len(train))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_subset = train.iloc[train_indices]
val_subset = train.iloc[val_indices]

# Prepare training data
train_data_dict = {
    'query': [f"{train_subset['answer'].values[x]} [SEP] question: {train_subset['question'].values[x]}" for x in range(len(train_subset)) for score in [0, 1, 2]],
    'response': [f"{json.loads(train_subset['rubric'].values[x])[str(score)]} [SEP] question: {train_subset['question'].values[x]}" for x in range(len(train_subset)) for score in [0, 1, 2]],
    'label': [1 if score == train_subset['level'].values[x] else 0 for x in range(len(train_subset)) for score in [0, 1, 2]]
}
train_dataset = Dataset.from_dict(train_data_dict)

# Prepare validation data
val_data_dict = {
    'query': [f"{val_subset['answer'].values[x]} [SEP] question: {val_subset['question'].values[x]}" for x in range(len(val_subset)) for score in [0, 1, 2]],
    'response': [f"{json.loads(val_subset['rubric'].values[x])[str(score)]} [SEP] question: {val_subset['question'].values[x]}" for x in range(len(val_subset)) for score in [0, 1, 2]],
    'label': [1 if score == val_subset['level'].values[x] else 0 for x in range(len(val_subset)) for score in [0, 1, 2]]
}
val_dataset = Dataset.from_dict(val_data_dict)

model = SentenceTransformer(train_config["model_name"], trust_remote_code=True)
loss = losses.CosineSimilarityLoss(model=model)

args = SentenceTransformerTrainingArguments(
    learning_rate=train_config['learning_rate'],
    optim='adamw_torch',
    prompts={
        'anchor': 'student answer: ',
        'positive': 'grading rubric: ',
        'negative': 'grading rubric: '
    },
    logging_steps=500,
    save_strategy='no',
    eval_strategy='steps',
    eval_steps=500,
    num_train_epochs=train_config['epochs'],
    warmup_steps=1000,
)

args.set_dataloader(train_batch_size=train_config["batch_size"], eval_batch_size=32)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss
)
trainer.train()
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
    f1 = f1_score(ref, pred, labels=[0, 1, 2], average='weighted')
    print(f1)
    
    qwk = cohen_kappa_score(ref, pred, labels=[0, 1, 2], weights='quadratic')
    print(qwk)
    print(confusion_matrix(ref, pred, labels=[0, 1, 2]))
    wandb.log({
        f'{dataset}_f1': f1,
        f'{dataset}_qwk': qwk
    })
    dddf = {
        'pred': pred,
        'ref': ref
    }
    pd.DataFrame.from_dict(dddf).to_csv(f'{dataset}_sbert_bert_results.csv')
model.save('./sbert-variant-6')