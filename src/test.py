import os
import json
import pandas as pd 
raw_fields = ["Id","EssaySet","score","EssayText"]
path_train = "data_raw/train_rel_2.tsv"
path_test = "data_raw/public_leaderboard_rel_2.tsv"
path_test_score = "data_raw/public_leaderboard_solution.csv"
train_df = pd.read_csv(path_train, sep="\t")
train_df.rename(columns={"Score1": "score"}, inplace=True)
train_df.drop(columns=["Score2"], inplace=True)
train_df.to_csv("data/train.csv", sep="\t", index=False)