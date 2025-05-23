import os
import json
import pandas as pd 
raw_fields = ["Id","EssaySet","score","EssayText"]
path_train = "data_raw/train.tsv"
path_test = "data_raw/public_leaderboard.tsv"
path_test_score = "data_raw/public_leaderboard_solution.csv"
train_df = pd.read_csv(path_train, sep="\t")
train_df = train_df.rename(columns={"Score1": "score"})
train_df = train_df.drop(columns=["Score2"])

test_df = pd.read_csv(path_test, sep="\t")
test_score_df = pd.read_csv(path_test_score)
test_score_ids = set(test_score_df["id"].unique())
test_df = test_df[test_df["Id"].isin(test_score_ids)]
test_df["score"] = test_score_df["essay_score"]
test_df.to_csv("data/test.csv", index=False)