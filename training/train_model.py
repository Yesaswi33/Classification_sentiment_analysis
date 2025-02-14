import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load processed data
with open("/Users/yesaswimadabattula/Documents/classification/processed_data.pkl", "rb") as f:
    df = pickle.load(f)

# TF-IDF Vectorization (Term frequency-inverse document frequency) used to classify text into numerical data
vectorizer = TfidfVectorizer(max_features=5000) #limit of 5000 features
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
#text classification tasks, such as spam detection, sentiment analysis, and document categorization
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model & vectorizer
with open("../models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)