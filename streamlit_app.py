
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="SpamShield | Email Detector",
    page_icon="📧",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {
    padding: 28px; border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #243b64);
    color: white; margin-bottom: 22px;
}
.hero p {color: #e0e7ff;}
.result {
    padding: 20px; border-radius: 14px;
    border: 1px solid rgba(128,128,128,.3);
}
div.stButton > button {
    border-radius: 10px; min-height: 44px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

st.markdown("""
<div class="hero">
    <h1>📧 SpamShield</h1>
    <p>AI-powered email and message classification</p>
    <p>Check suspicious messages before trusting them.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Could not load the model or vectorizer.")
    st.code(str(e))
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("Machine learning email classification.")
    st.write("Model: Multinomial Naive Bayes")
    st.write("Features: CountVectorizer")
    st.caption("Confidence is a model estimate, not a guarantee.")

spam_example = (
    "Congratulations! You have won a FREE prize. "
    "Claim your cash reward now by clicking this link!"
)
ham_example = (
    "Hi, are we still meeting tomorrow at college? "
    "Please let me know what time works for you."
)

left, right = st.columns(2)

with left:
    st.subheader("📝 Enter a message")
    if st.button("Load spam example", use_container_width=True):
        st.session_state.email_text = spam_example

with right:
    st.subheader("🧪 Try a sample")
    if st.button("Load legitimate example", use_container_width=True):
        st.session_state.email_text = ham_example

email = st.text_area(
    "Email or SMS content",
    key="email_text",
    height=190,
    placeholder="Paste the full message here..."
)

if st.button("🔍 Analyze Message", type="primary",
             use_container_width=True):
    if not email.strip():
        st.warning("Enter a message or load one of the examples.")
    else:
        try:
            X = vectorizer.transform([email])
            prediction = model.predict(X)[0]

            classes = list(model.classes_)
            spam_labels = {"1", "spam", "true"}
            is_spam = str(prediction).strip().lower() in spam_labels

            st.divider()
            st.subheader("Analysis result")

            if is_spam:
                st.error("🚨 SPAM DETECTED")
                st.write("The model classifies this message as spam.")
            else:
                st.success("✅ LIKELY LEGITIMATE")
                st.write("The model classifies this message as not spam.")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                confidence = max(probs) * 100

                c1, c2 = st.columns(2)
                c1.metric("Prediction confidence",
                          f"{confidence:.1f}%")
                c2.metric("Model prediction", 
                          "Spam" if is_spam else "Not Spam")

                st.caption(
                    "This is the model's estimated probability "
                    "for its selected class, not measured test accuracy."
                )

                score_data = pd.DataFrame({
                    "Class": [str(c) for c in classes],
                    "Estimated probability (%)": [
                        float(p) * 100 for p in probs
                    ]
                })
                st.bar_chart(
                    score_data.set_index("Class")
                )
            else:
                st.info(
                    "This model does not provide predict_proba(). "
                    "Confidence is not available."
                )

        except Exception as e:
            st.error("Prediction failed. Check your model's label mapping.")
            st.code(str(e))

st.divider()
st.caption("SpamShield • Machine Learning Project")
