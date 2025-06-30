import pandas as pd
import json
import random
import os 
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, losses
import wandb
from datasets import Dataset
wandb.login(key=os.getenv('WANDB_API_KEY'))
train_config = {
    'model_name': 'deepset/gbert-large',
    'learning_rate': 1e-5,
    'batch_size': 16,
    'epochs': 5,
    'dataset': 'ALICE'}
wandb.init(
    project='alice-rubrics',
    name='crossenc-gbert-sg',
    config=train_config) 
train = pd.read_csv('alice_data/ALICE_train_new.csv')
# Shuffle and split the data for validation (80% train, 20% validation)
train_indices = list(range(len(train)))
random.shuffle(train_indices)
train_split = int(0.8 * len(train))
train_idx = train_indices[:train_split]
val_idx = train_indices[train_split:]

# Create training dataset
train_data_dict = {
    'query': [f"{train['answer'].values[x]}" for x in train_idx for score in [0, 1, 2]],
    'response': [f"{json.loads(train['rubric'].values[x])[str(score)]} [SEP] question: {train['question'].values[x]}" for x in train_idx for score in [0, 1, 2]],
    'label': [1 if score == train['level'].values[x] else 0 for x in train_idx for score in [0, 1, 2]]
}
train_dataset = Dataset.from_dict(train_data_dict)
# Create validation dataset
val_data_dict = {
    'query': [f"{train['answer'].values[x]}" for x in val_idx for score in [0, 1, 2]],
    'response': [f"{json.loads(train['rubric'].values[x])[str(score)]} [SEP] question: {train['question'].values[x]}" for x in val_idx for score in [0, 1, 2]],
    'label': [1 if score == train['level'].values[x] else 0 for x in val_idx for score in [0, 1, 2]]
}
val_dataset = Dataset.from_dict(val_data_dict)

model = CrossEncoder(train_config["model_name"], num_labels=1)
loss = losses.BinaryCrossEntropyLoss(model=model)

args = CrossEncoderTrainingArguments(
    learning_rate=train_config['learning_rate'],
    optim='adamw_torch',
    prompts={
        'query': 'student answer: ',
        'response': 'grading rubric: ',
    },
    logging_steps=50,
    save_strategy='steps',
    save_steps=1500,
    eval_strategy='steps',
    eval_steps=1000,
    report_to=['wandb'],
    num_train_epochs=train_config['epochs'],
    warmup_steps=1000,
)
args.set_dataloader(train_batch_size=train_config["batch_size"], eval_batch_size=32)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss
)
trainer.train()
model.save('./crossenc-variant-1')


model = CrossEncoder("./crossenc-variant-1")

for dataset in ['UA', 'UQ']:
    test_name = "test_" + dataset.lower()
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
    f1 = f1_score(ref, pred, labels=[0, 1, 2], average='weighted')
    qwk = cohen_kappa_score(ref, pred, labels=[0, 1, 2], weights='quadratic')
    print(f"F1 Score: {f1}")
    print(f"Quadratic Weighted Kappa: {qwk}")
    report = {test_name: {
        'f1': f1,
        'qwk': qwk
    }}
    # Log metrics to wandb
    wandb.log(report)
    print(confusion_matrix(ref, pred, labels=[0, 1, 2]))
    

    dddf = {
        'pred': pred,
        'ref': ref
    }

    pd.DataFrame.from_dict(dddf).to_csv(f'{dataset}_crossenc_bert_results.csv')