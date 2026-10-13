import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

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

    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text


data["clean_text"] = data["content"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_text"])

y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully")
print("Accuracy:", accuracy)
print("Precision:", precision_score(y_test, predictions))
print("F1-Score:", f1_score(y_test, predictions))
print("Recall Score:", recall_score(y_test, predictions))
