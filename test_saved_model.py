import re
import joblib
import pandas as pd

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

model = joblib.load("best_model.pkl")

vectorizer = joblib.load("comparison_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


article = input(
    "WASHINGTON (Reuters) - The Pentagon has notified the U.S. Congress of the sale to Nigeria of 12 Super Tucano A-29 planes and weapons worth $593 million, which the West African country wants for its fight against the militant group Boko Haram. The Federal Register on Monday published the Aug. 2 notification from the Pentagon’s Defense Security Cooperation Agency. The sale includes thousands of bombs and rockets and was originally agreed by former President Barack Obama’s administration. The Super Tucano A-29, an agile, propeller-driven plane with reconnaissance and surveillance as well as attack capabilities, is made by Brazil’s Embraer. A second production line is in Florida, in a partnership between Embraer and privately held Sierra Nevada Corp of Sparks, Nevada."
)

cleaned = clean_text(article)

vector = vectorizer.transform([cleaned])
prediction = model.predict(vector)[0]

print("Prediction:", "Real News" if prediction == 1 else "Fake News")

# print("\nRandom Fake article:\n")
# print(fake.sample(1)["text"].values[0][:800])

# print("\nRandom Real article:\n")
# print(real.sample(1)["text"].values[0][:800])
