import streamlit as st
import joblib
import matplotlib.pyplot as plt

model = joblib.load("factor_model.joblib")

st.title("Customer Review Analyzer")

rating = st.slider("Rating",1,5)

comment = st.text_area("Write your review")

if st.button("Analyze"):

    factor = model.predict([comment])[0]

    if rating <= 2:
        sentiment = "Negative"
    elif rating == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    st.write("Sentiment:", sentiment)
    st.write("Factor:", factor)

    labels = ["Billing","Staff Behavior","Cleanliness",
              "Product Availability","Pricing","Queue / Waiting Time","Others"]

    values = [1 if f==factor else 0 for f in labels]

    fig, ax = plt.subplots()
    ax.bar(labels,values)
    ax.set_xticklabels(labels, rotation=40)

    st.pyplot(fig)
    