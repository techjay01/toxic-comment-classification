import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("train.csv")

# Create binary label
df["toxic_label"] = (
    df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]
    .sum(axis=1)
    .apply(lambda x: 1 if x > 0 else 0)
)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["comment_text"] = df["comment_text"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["comment_text"], df["toxic_label"], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "toxic_comment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
