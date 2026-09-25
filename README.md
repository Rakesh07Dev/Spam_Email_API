# 📧 Spam Email Detection API

A machine learning-based API that classifies text messages as **Spam** or **Not Spam (Ham)**.

The project uses **Multinomial Naive Bayes** and **CountVectorizer** to analyze text and make predictions.

## 🚀 Features

- Detects whether a message is Spam or Not Spam
- Machine learning-based classification
- FastAPI backend
- Text feature extraction using CountVectorizer
- Pre-trained model included
- Simple API endpoint for predictions

## 🛠️ Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- Uvicorn
- Joblib
- Multinomial Naive Bayes
- CountVectorizer

## 📂 Project Structure

```text
Spam_Email_API/
│
├── app.py                 # FastAPI application
├── train_model.py         # Model training script
├── spam.csv               # Dataset
├── model.pkl              # Trained machine learning model
├── vectorizer.pkl         # Trained text vectorizer
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
