import streamlit as st
import joblib
import re

# Load model
model = joblib.load("toxic_comment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛑")

st.title("🛑 Toxic Comment Classification")
st.write("Detect whether a comment is **toxic or non-toxic**.")

comment = st.text_area("Enter a comment")

if st.button("Analyze"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(comment)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")
