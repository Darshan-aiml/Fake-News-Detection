from google.colab import files

# Upload both files: Fake.csv and True.csv
uploaded = files.upload()

import os

print(os.listdir())

import pandas as pd

fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')

print(fake_df.head())
print(real_df.head())

# Add labels
fake_df['label'] = 0  # Fake
real_df['label'] = 1  # Real

# Combine and shuffle the data
data = pd.concat([fake_df, real_df])
data = data.sample(frac=1).reset_index(drop=True)

# Keep only title and label columns
data = data[['title', 'label']]
data.head()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Features and labels
X = data['title']
y = data['label']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_news(news_text):
    news_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(news_tfidf)
    label = "REAL" if prediction[0] == 1 else "FAKE"
    return label

# Simulate user input
news_text = input("Enter a news headline to check if it's fake or real: ")
result = predict_news(news_text)
print("Prediction:", result)

!pip install gradio

import gradio as gr

def predict_news(news_text):
    news_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(news_tfidf)
    label = "REAL" if prediction[0] == 1 else "FAKE"
    return f"This news is: {label}"

gr.Interface(fn=predict_news, inputs="text", outputs="text", title="Fake News Detector").launch()

import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
