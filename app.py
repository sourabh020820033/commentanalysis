import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("toxic_comment_model.pkl")

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# -----------------------------
# Text preprocessing
# -----------------------------
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def stemming(text):
    return " ".join(stemmer.stem(word) for word in text.split())

def preprocess(text):
    return stemming(clean_text(text))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")

st.title("🚨 Toxic Comment Classification (Binary Output)")
st.write("1 = Toxic detected | 0 = Not detected")

user_input = st.text_area("💬 Enter comment:", height=120)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        processed_text = preprocess(user_input)

        # Binary predictions
        predictions = model.predict([processed_text])[0]

        # -----------------------------
        # Show table
        # -----------------------------
        result_df = pd.DataFrame({
            "Label": labels,
            "Prediction": predictions
        })

        st.subheader("📋 Prediction Table")
        st.dataframe(result_df, use_container_width=True)

        # -----------------------------
        # Graph
        # -----------------------------
        st.subheader("📊 Toxic Label Graph (0 / 1)")

        fig, ax = plt.subplots()
        ax.bar(result_df["Label"], result_df["Prediction"])
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Prediction (0 or 1)")
        ax.set_xlabel("Toxic Labels")
        ax.set_title("Toxic Comment Classification Result")
        plt.xticks(rotation=30)

        st.pyplot(fig)