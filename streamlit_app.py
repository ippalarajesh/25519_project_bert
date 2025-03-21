# The Streamlit app allows users to interact with the trained BERT model via a web interface.

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./model/trained_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

# Streamlit UI
st.title("BERT Sentiment Classification")
st.write("Enter a sentence to predict its sentiment")

input_text = st.text_area("Text input", "Enter text here...")
if st.button("Classify"):
    sentiment = predict_sentiment(input_text)
    st.write(f"Sentiment: {sentiment}")
