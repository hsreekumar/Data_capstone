import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import re
import os
import subprocess
import sys

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('nltk')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split



# Preprocessing functions
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Replace non-alphabet characters with spaces
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    return text

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

def stem_and_lemmatize(words):
    stemmed = [ps.stem(word) for word in words]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return lemmatized

def preprocess_headline(headline):
    cleaned = clean_text(headline)
    tokens = tokenize_and_remove_stopwords(cleaned)
    normalized = stem_and_lemmatize(tokens)
    return " ".join(normalized)

# Load data
input_path = '/opt/ml/processing/input/Combined_News_DJIA.csv'
df = pd.read_csv(input_path, encoding='ISO-8859-1')

# Split into train and test sets
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Preprocess the headlines
train_headlines = []
for row in range(0, len(train.index)):
    combined_headlines = ' '.join(str(x) for x in train.iloc[row, 2:27])
    preprocessed_headline = preprocess_headline(combined_headlines)
    train_headlines.append(preprocessed_headline)

test_headlines = []
for row in range(0, len(test.index)):
    combined_headlines = ' '.join(str(x) for x in test.iloc[row, 2:27])
    preprocessed_headline = preprocess_headline(combined_headlines)
    test_headlines.append(preprocessed_headline)

# Save preprocessed data
output_train_path = '/opt/ml/processing/train/train.csv'
output_validation_path = '/opt/ml/processing/validation/validation.csv'
output_test_path = '/opt/ml/processing/test/test.csv'

train_data = pd.DataFrame({'headline': train_headlines, 'label': train['Label'].values})
test_data = pd.DataFrame({'headline': test_headlines, 'label': test['Label'].values})

train_data.to_csv(output_train_path, index=False)
test_data.to_csv(output_validation_path, index=False)
test_data.to_csv(output_test_path, index=False)