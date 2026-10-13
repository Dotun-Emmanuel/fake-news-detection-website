import pandas as pd

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data["content"] = data["title"] + " " + data["text"]

print(data["content"].head())
print(data["label"].value_counts())