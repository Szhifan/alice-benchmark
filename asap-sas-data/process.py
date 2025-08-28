import pandas as pd 
import json 

path_train = "asap-sas-data/train.csv"
path_test = "asap-sas-data/test.csv"
rubrics = {}
for i in range(1,11):
    path_rub = f"asap-sas-data/rubrics/set{i}.json"
    with open(path_rub, "r") as f:
        rubrics[i] = json.load(f)["rubrics"]


df_train = pd.read_csv(path_train)
df_train_2 = df_train.copy()
df_train["answer"] = df_train_2["level"]
df_train["level"] = df_train_2["answer"]
df_train.to_csv(path_train, index=False)