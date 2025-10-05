# movie_genre_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Example CSV format:
# | plot_summary                              | genre     |
# |-------------------------------------------|-----------|
# | "A young boy discovers he is a wizard..." | Fantasy   |
# | "A detective solves a mysterious crime"   | Thriller  |

# Replace with your dataset file path
df = pd.read_csv("movies.csv")

print("Dataset Sample:")
print(df.head())

# -----------------------------
# 2. Split into train & test
# -----------------------------
X = df["plot_summary"]   # input text
y = df["genre"]          # output label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Text Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 4. Train Classifier (Logistic Regression)
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Test with new plot summaries
# -----------------------------
while True:
    user_input = input("\nEnter a movie plot summary (or 'quit' to exit): ")
    if user_input.lower() == "quit":
        break
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]
    print("Predicted Genre:", prediction)
