import pandas as pd
import re

# Read Dataset

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add Labels
fake["label"] = 0
real["label"] = 1

# Combine Dataset

data = pd.concat([fake, real])

# One content column

data["content"] = data["title"] + " " + data["text"]


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\\S+","", text)
    text = re.sub(r"[^a-zA-Z ]","", text)

    return text

data["clean_text"] = data["content"].apply(clean_text)


print(data[["content","clean_text"]].head())
