import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("factor_model.joblib")

st.title("Customer Feedback Analyzer")

# Rating input
rating = st.slider("Give Rating",1,5)

# Comment input
comment = st.text_area("Write your comment")

# Satisfaction selection
satisfaction = st.selectbox(
    "How satisfied are you?",
    ["Positive (Satisfied) 🤩","Neutral 😐","Negative (Dissatisfied) ☹️"]
)

if st.button("Analyze"):

    # Factor prediction
    factor = model.predict([comment])[0]

    # Sentiment from rating
    if rating <= 2:
        sentiment = "Negative"
    elif rating == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    st.subheader("Prediction Result")

    st.write("Sentiment:", sentiment)
    st.write("Factor:", factor)

    # -------- FACTOR GRAPH --------
    labels = [
        "Billing",
        "Staff Behavior",
        "Cleanliness",
        "Product Availability",
        "Pricing",
        "Queue / Waiting Time",
        "Others"
    ]

    values = [1 if f == factor else 0 for f in labels]

    fig1, ax1 = plt.subplots()
    ax1.bar(labels, values)
    ax1.set_title("Issue Category")
    ax1.set_xticklabels(labels, rotation=45)

    st.pyplot(fig1)

    # -------- SATISFACTION GRAPH --------
    sat_labels = ["Positive 🤩","Neutral 😐","Negative ☹️"]

    if "Positive" in satisfaction:
        sat_values = [1,0,0]
    elif "Neutral" in satisfaction:
        sat_values = [0,1,0]
    else:
        sat_values = [0,0,1]

    fig2, ax2 = plt.subplots()
    ax2.bar(sat_labels, sat_values)
    ax2.set_title("Customer Satisfaction Level")

    st.pyplot(fig2)