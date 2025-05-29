# 📰 Fake News Detection using Machine Learning

This project is a simple yet effective **Fake News Detector** that uses **Logistic Regression** and **TF-IDF vectorization** to classify news headlines as **REAL** or **FAKE**. It includes a web-based UI built with **Gradio** for easy testing.

---

## 🚀 Features

- Detects fake vs. real news headlines
- Built with `scikit-learn` and `pandas`
- Interactive UI using **Gradio**
- Trained on real-world data (Fake.csv & True.csv)
- Runs entirely in **Google Colab** or Python locally

---

## 📁 Dataset

The dataset consists of two CSV files:
- `Fake.csv`: Contains fake news headlines
- `True.csv`: Contains real news headlines

Each entry includes a news title and label:
- `0` = Fake
- `1` = Real

---

## 🧠 Model Used

- **TF-IDF Vectorizer**: Converts headlines into numerical features
- **Logistic Regression**: A lightweight but powerful model for binary classification

---

## 🖥️ Gradio Web App

You can try the model live using a simple Gradio interface.  
Paste your own headline and check if it’s predicted as **REAL** or **FAKE**!

👉  `https://1d664f075e389c66d7.gradio.live/`

---

