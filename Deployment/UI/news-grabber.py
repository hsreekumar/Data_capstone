import streamlit as st
import requests
from datetime import datetime

# Function to fetch news data
def fetch_news(date):
    url = f"https://newsapi.org/v2/everything?sources=cnn&from={date}&to={date}&language=en&pageSize=25&sortBy=popularity&apiKey=9034a4796d7d4e4d94ebf59de07a6b5a"
    response = requests.get(url)
    return response.json()

# Function to extract titles from news articles
def extract_titles(articles):
    titles = [article['title'] for article in articles]
    combined_titles = " ||| ".join(titles)
    return combined_titles

# Streamlit app UI
st.title("News Titles Fetcher")
st.write("Enter a date to fetch news titles from CNN:")

# Date input
date_input = st.date_input("Select a date", value=datetime.today())

# Convert date to string format
date_str = date_input.strftime("%Y-%m-%d")

# Fetch and display news titles when the button is clicked
if st.button("Fetch News Titles"):
    news_data = fetch_news(date_str)
    
    # Check if the API returned articles
    if 'articles' in news_data:
        titles = extract_titles(news_data['articles'])
        st.write(f"Combined Titles: {titles}")
    else:
        st.write("No news articles found for the selected date.")