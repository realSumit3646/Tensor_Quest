#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Function to load and preprocess the model
@st.cache_resource
def load_preprocessed_data():
    df = pd.read_csv(r"Spam_SMS.csv")

    # Label encoding and preprocessing
    le = LabelEncoder()
    df["Class_Label"] = le.fit_transform(df["Class"])
    df['processed_text'] = df['Message']
    df.drop(["Message", "Class"], axis=1, inplace=True)

    # Tokenization and padding
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n', num_words=1000)
    tokenizer.fit_on_texts(df['processed_text'])

    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    return tokenizer, max_len, le

# Load the trained model and data preprocessing functions
@st.cache_resource
def load_model_and_dependencies():
    model = load_model('spam_classifier_model.h5')
    tokenizer, max_len, le = load_preprocessed_data()
    return model, tokenizer, max_len, le

model, tokenizer, max_len, le = load_model_and_dependencies()

# Streamlit UI
st.markdown("""
    <style>
        .main {
            background-color: #232A5A;
            color: #333;
            font-family: 'Helvetica', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            font-weight: bold;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            border-radius: 12px;
        }
        .stTextArea>textarea {
            font-size: 16px;
            padding: 10px;
            border-radius: 5px;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("Spam SMS Detection")
st.write("Enter an SMS message below to check if it is spam or not.")

# Input for SMS message
input_text = st.text_area("Enter SMS text")

# Function to predict whether the text is spam or not
def predict_spam(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')

    prediction = model.predict(input_padded)
    predicted_class = np.argmax(prediction, axis=1)

    if predicted_class == 1:
        return "Spam"
    else:
        return "Not Spam"

# Add an Enter button
if st.button('Predict'):
    if input_text:
        result = predict_spam(input_text)
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a message to check.")

st.markdown('</div>', unsafe_allow_html=True)

