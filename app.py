import streamlit as st
import joblib
import sys
import os

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Emotion Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# ------------------------------
# Import preprocessing function
# ------------------------------

sys.path.append("src")
from preprocessing import clean_text

# ------------------------------
# Load Models
# ------------------------------

model = joblib.load("models/emotion_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ------------------------------
# Custom CSS
# ------------------------------

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.title{
    text-align:center;
    color:#2E86C1;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.title("Project Information")

st.sidebar.success("Model : Logistic Regression")

st.sidebar.success("Vectorizer : TF-IDF")

st.sidebar.success("Accuracy : 94.36 %")

st.sidebar.markdown("---")

st.sidebar.write("Developer")

st.sidebar.info("Akash Chauhan")

# ------------------------------
# Main Heading
# ------------------------------

st.markdown('<p class="title">😊 Emotion Sentiment Analysis</p>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Predict human emotions from text using NLP & Machine Learning.</p>',
unsafe_allow_html=True)

st.write("")

# ------------------------------
# Input Text
# ------------------------------

user_input = st.text_area(
    "Enter your text here",
    height=180,
    placeholder="Example: I am feeling very happy today..."
)

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Emotion", use_container_width=True):

    if user_input.strip() == "":

        st.warning("⚠ Please enter some text.")

    else:

        cleaned = clean_text(user_input)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        emotion = label_encoder.inverse_transform(prediction)[0]

        if emotion.lower() == "joy":

            st.success(f"😊 Predicted Emotion : {emotion.title()}")

        elif emotion.lower() == "anger":

            st.error(f"😠 Predicted Emotion : {emotion.title()}")

        elif emotion.lower() == "fear":

            st.warning(f"😨 Predicted Emotion : {emotion.title()}")

        else:

            st.info(f"Prediction : {emotion}")

        st.markdown("---")

        st.subheader("Processed Text")

        st.code(cleaned)

# ------------------------------
# Sample Inputs
# ------------------------------

st.markdown("---")

st.subheader("Sample Inputs")

st.code("I am very happy today.")

st.code("I am scared to go outside.")

st.code("I am extremely angry right now.")

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")

st.markdown(
'<p class="footer">Developed with ❤️ by <b>Akash Chauhan</b></p>',
unsafe_allow_html=True)