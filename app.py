from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load Model
model = joblib.load("model.pkl")

# Load Vectorizer
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()


class Email(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Spam Email Detection API is Live!",
        "docs": "/docs"
    }


@app.post("/predict")
def predict(email: Email):

    # Convert text into numbers
    text = vectorizer.transform([email.message])

    # Prediction
    prediction = model.predict(text)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return {
        "prediction": int(prediction),
        "result": result
    }