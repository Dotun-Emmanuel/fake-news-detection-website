import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# Load the datasets
fake_news = pd.read_csv("Fake.csv")
real_news = pd.read_csv("True.csv")


# Add labels
# 0 means fake news
# 1 means real news
fake_news["label"] = 0
real_news["label"] = 1


# Merge the two datasets
data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)


# Replace missing title or text values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")


# Combine each article's title and text
data["content"] = data["title"] + " " + data["text"]


# Remove duplicate and empty articles
data = data.drop_duplicates(subset=["content"])
data = data[data["content"].str.strip() != ""]


# Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Cleaning the dataset...")

data["clean_text"] = data["content"].apply(clean_text)


# Prepare the input text and labels
X = data["clean_text"]
y = data["label"]


# Split the raw text into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)


# Fit TF-IDF on the training data only
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data with the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)


# Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}


# Store the model results
results = []

best_model = None
best_model_name = ""
best_f1_score = 0


# Train and test each model
for model_name, model in models.items():

    print("\nTraining", model_name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix")
    print(matrix)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Fake News", "Real News"]
    )

    display.plot(values_format="d")
    plt.title("Confusion Matrix: " + model_name)
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        filename + "_confusion_matrix.png",
        dpi=300,
        bbox_inches ="tight"
    )

    plt.close()
 
    with open(
        filename + "_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model
        best_model_name = model_name


# Create and display the comparison table
results_table = pd.DataFrame(results)

results_table = results_table.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nModel Comparison")

print(
    results_table.to_string(
        index=False
    )
)


# Save the comparison results
results_table.to_csv(
    "model_comparison_results.csv",
    index=False
)


# Save the best-performing model
joblib.dump(
    best_model,
    "best_model.pkl"
)


# Save the vectorizer used with the best model
joblib.dump(
    vectorizer,
    "comparison_vectorizer.pkl"
)


print("\nBest model:", best_model_name)
print(f"Best F1 Score: {best_f1_score:.4f}")

print("\nSaved files:")
print("best_model.pkl")
print("comparison_vectorizer.pkl")
print("model_comparison_results.csv")