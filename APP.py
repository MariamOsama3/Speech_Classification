# APP.py
import streamlit as st
import pickle
import numpy as np
import sys
import os
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import from utils
try:
    from utils.preprocessing import clean_tweet
except ImportError:
    def clean_tweet(text):
        # Fallback simple cleaning if preprocessing.py is missing
        return text.lower().strip()

# Configure page
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput>div>div>input {
        font-size: 18px !important;
        padding: 12px !important;
    }
    .stProgress>div>div>div>div {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('model10.h5')  # Use your existing model file
    try:
        with open('tokenizer.pkl', 'rb') as f:  # Load from root directory
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Tokenizer file (tokenizer.pkl) not found!")
        raise
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# App Header
st.title("🐦 Tweet Sentiment Analyzer")
st.markdown("""
Classify tweets as **Positive 😊** or **Negative 😠** using Bidirectional LSTM
""")

# Input Section
user_input = st.text_area(
    "Enter your tweet:",
    "I love this product!",
    height=150
)

# Prediction Logic
def predict_sentiment(text):
    cleaned_text = clean_tweet(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    return cleaned_text, prediction

# Button Action
if st.button("Analyze Sentiment", type="primary"):
    with st.spinner("Processing..."):
        cleaned_text, prob = predict_sentiment(user_input)
        
        # Display Results
        st.subheader("🔍 Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Sentiment", 
                     value="😊 Positive" if prob < 0.5 else "😠 Negative")
        with col2:
            st.metric("Confidence", 
                     value=f"{max(prob, 1-prob)*100:.1f}%")
        
        # Confidence Bar
        confidence = abs(prob - 0.5) * 2  # Convert to 0-1 scale
        st.progress(confidence)
        
        # Show cleaned text
        with st.expander("See cleaned text"):
            st.code(cleaned_text)

# Sample Tweets
st.markdown("---")
st.subheader("💡 Try these examples:")
examples = st.columns(3)
with examples[0]:
    if st.button("Positive 😊", help="Example: I love this!"):
        st.session_state.user_input = "The service was amazing! Will definitely return."
with examples[1]:
    if st.button("Neutral 😐", help="Example: It's okay"):
        st.session_state.user_input = "The food was average, nothing special."
with examples[2]:
    if st.button("Negative 😠", help="Example: I hate this"):
        st.session_state.user_input = "Worst customer service ever!"