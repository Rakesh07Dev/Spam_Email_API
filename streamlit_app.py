
import streamlit as st
import joblib

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("📧 Spam Email Detector")
st.write("Enter an email message to check for spam.")

message = st.text_area("Email message", height=180)

if st.button("Detect Spam", type="primary"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]

        # Assumes label 1 means spam
        if str(prediction).lower() in ["1", "spam", "true"]:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ NOT SPAM")
