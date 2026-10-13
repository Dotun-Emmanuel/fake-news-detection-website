import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

project_folder = Path(__file__).resolve().parent

model = joblib.load(project_folder / "best_model.pkl")
vectorizer = joblib.load(
    project_folder / "comparison_vectorizer.pkl"
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please submit valid article data."}), 400

    article = data.get("news_text")

    if not isinstance(article, str) or not article.strip():
        return jsonify({"error": "Please enter a news article."}), 400

    cleaned_article = clean_text(article)

    if not cleaned_article:
        return jsonify({"error": "Please enter valid article text."}), 400

    article_vector = vectorizer.transform([cleaned_article])
    prediction = int(model.predict(article_vector)[0])

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run()