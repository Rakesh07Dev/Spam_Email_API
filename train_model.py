import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Features and Target
X = df["message"]
y = df["label"]

# Convert text into numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("Accuracy :", accuracy)

# Save Model
joblib.dump(model, "model.pkl")

# Save Vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")
print("Vectorizer Saved Successfully!")

