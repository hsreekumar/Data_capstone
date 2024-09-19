import streamlit as st
import json
import requests
from datetime import datetime

# Title of the app
st.title("News Headlines Input for Sentiment Analysis")

# Initialize the input structure
data = []

# Number of dates input
num_dates = st.number_input("How many dates do you want to input?", min_value=1, step=1, value=1)

# Iterate over each date
for i in range(num_dates):
    st.subheader(f"Input for Date {i+1}")
    date = st.date_input(f"Select Date {i+1}", datetime.today(), key=f"date_{i}")
    date_str = date.strftime('%Y-%m-%d')

    # Input for multiple headlines for each date
    st.write("Please use '|||' as a separator for headlines.")
    headlines_input = st.text_area(f"Enter headlines for {date_str}, separated by '|||'", key=f"headlines_{i}")

    # Split the input headlines into a list using '|||' as the separator
    headlines = [headline.strip() for headline in headlines_input.split("|||") if headline.strip()]

    # Store the data in the required format
    if headlines:
        data.append({"date": date_str, "headlines": headlines})

# JSON formatting and API call directly on 'Predict' button click
if st.button("Predict"):
    if data:
        json_data = {"data": data}
        st.subheader("Formatted JSON")
        st.json(json_data)
        
        # Send the data to the API Gateway
        url = "https://t7k6uzy7e9.execute-api.us-east-1.amazonaws.com/prod/news"
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=json_data, headers=headers)

        if response.status_code == 200:
            st.success("Prediction successful!")
            st.json(response.json())  # Display the response
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please input headlines for the dates.")