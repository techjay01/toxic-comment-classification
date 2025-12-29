import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("toxic_comment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Overall background and text */
    .stApp {
        background-color: #f9f9fb;
        color: #0e1117;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Text area styling */
    textarea {
        border-radius: 12px;
        border: 1px solid #ccc;
        padding: 12px;
        font-size: 16px;
        background-color: #ffffff;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }

    /* Predict button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 25px;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #45a049;
        cursor: pointer;
    }

    /* Result headers */
    h2 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
    }

    /* Footer styling */
    footer {
        color: #666;
        font-size: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🛡️ Toxic Comment Classification")
st.markdown("""
Welcome! This app predicts whether a comment is **Toxic** or **Non-Toxic** using a Logistic Regression machine learning model.
Enter any comment below and click **Predict** to see the result.
""")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Type or paste a comment in the text box.
2. Click **Predict**.
3. See if the comment is classified as **Toxic** or **Non-Toxic**.
""")

# Input from user
user_input = st.text_area("Enter your comment here:", height=180)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment before predicting.")
    else:
        # Vectorize input
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        confidence = np.max(model.predict_proba(vect_input)) * 100

        # Display results
        if prediction == 1:
            st.markdown(f"<h2 style='color:#e74c3c;'>Toxic ⚠️</h2>", unsafe_allow_html=True)
            st.markdown(f"Confidence: {confidence:.2f}%")
        else:
            st.markdown(f"<h2 style='color:#2ecc71;'>Non-Toxic ✅</h2>", unsafe_allow_html=True)
            st.markdown(f"Confidence: {confidence:.2f}%")

# Footer
st.markdown("---")
st.markdown("Made by Group 15 | Dataset: [Kaggle Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)")
