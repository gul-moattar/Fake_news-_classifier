import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords if not already
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection Web App")
st.write("Enter a news article or headline to predict whether it's **Fake** or **Real**.")

# Text input
user_input = st.text_area("Enter News Text", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text first.")
    else:
        # Preprocess
        cleaned_text = preprocess_text(user_input)
        vec_input = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(vec_input)[0]
        confidence = model.predict_proba(vec_input)[0]

        label_map = {0: 'Fake', 1: 'Real'}
        result = label_map[prediction]
        score = max(confidence)

        # Display results
        st.success(f"🧾 Prediction: {result}")
        st.info(f"🔍 Confidence Score: {score:.2f}")
