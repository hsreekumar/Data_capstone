import os
import shutil
import string

import re
import torch
import nltk
import logging
import boto3
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import shutil
import torch.nn as nn
from nltk.tokenize import word_tokenize
import numpy as np

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def cleanup_tmp_directory():
    tmp_dir = '/tmp'
    logger.info("Cleaning up /tmp directory")
    # Remove all files and subdirectories in /tmp
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    logger.info("Cleanup successful.")
    
    
cleanup_tmp_directory()

# Download the required resource
nltk.download('punkt', download_dir='/tmp')
nltk.download('punkt_tab', download_dir='/tmp')
nltk.download('stopwords', download_dir='/tmp')
nltk.download('wordnet', download_dir='/tmp')

# Update NLTK data path
nltk.data.path.append('/tmp')

# Check if NLTK resources are in /tmp
logger.info(f'NLTK data path: {nltk.data.path}')




stop_words = set(nltk.corpus.stopwords.words('english'))
ps = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
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

def preprocess_input(input_data):
    preprocessed_data = []
    for entry in input_data:
        date = entry['date']
        headlines = entry['headlines']
        
        combined_headlines = ' '.join(headlines)
        preprocessed_headline = preprocess_headline(combined_headlines)
        
        preprocessed_data.append({
            'date': date,
            'preprocessed_headline': preprocessed_headline
        })
    return preprocessed_data

class NewsDataset(Dataset):
    def __init__(self, headlines, tokenizer, max_len):
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, item):
        headline = str(self.headlines[item])

        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_data_loader(headlines, tokenizer, max_len, batch_size):
    ds = NewsDataset(
        headlines=headlines,
        tokenizer=tokenizer,
        max_len=max_len
    )

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=SequentialSampler(ds),
        num_workers=0  # Set to 0 to avoid multiprocessing
    )

    return data_loader
   
class NewsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.pooler_output
        attentions = outputs.attentions  # Extract attention weights
        
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attentions  # Return both logits and attention weights

def load_model(model_path):
    try:
        # Initialize the model and wrap with DataParallel
        model = NewsClassifier(n_classes=2)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error in loading model: {str(e)}")
        raise


def get_top_n_words(input_ids, attention_weights, tokenizer, n=10):
    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    logger.info(f'Tokens: {tokens}')
    
    # Ensure attention_weights is a 1D array or reduce to a 1D representation (take the mean in case of arrays)
    attention_weights = attention_weights.cpu().numpy()
    logger.info(f'Attention Weights: {attention_weights}')
    
    # Check if the attention weights are multidimensional
    if attention_weights.ndim > 1:
        logger.warning(f"Attention weights have more than one dimension. Reducing by taking the mean across dimensions.")
        attention_weights = np.mean(attention_weights, axis=-1)
        logger.info(f"Reduced Attention Weights: {attention_weights}")

    # Make sure that attention_weights has the same length as tokens
    if len(tokens) != len(attention_weights):
        logger.error(f"Mismatch between tokens and attention weights: Tokens length: {len(tokens)}, Attention weights length: {len(attention_weights)}")
        raise ValueError("Length of tokens and attention weights must be the same")

    # Pair tokens with their attention weights
    token_attention_pairs = list(zip(tokens, attention_weights))
    logger.info(f'Token-Attention Pairs: {token_attention_pairs}')
    
    # Filter out special tokens (those enclosed in square brackets) and tokens shorter than 3 characters
    token_attention_pairs = [(token, weight) for token, weight in token_attention_pairs if not (token.startswith('[') and token.endswith(']')) and not (token.startswith('#')) and len(token) > 3]
    logger.info(f'Token-Attention Pairs after filtering special tokens and short words: {token_attention_pairs}')
    
    # Sort tokens by attention weight (highest first)
    try:
        sorted_pairs = sorted(token_attention_pairs, key=lambda x: float(x[1]), reverse=True)
    except Exception as e:
        logger.error(f"Error during sorting: {e}")
        raise

    # Extract top N tokens and their weights
    top_n_tokens_with_weights = sorted_pairs[:n]
    top_n_tokens = [token for token, weight in top_n_tokens_with_weights]

    # Log the top N tokens with their weights
    logger.info(f'Top {n} Tokens with Weights: {top_n_tokens_with_weights}')
    
    return top_n_tokens


def lambda_handler(event, context):
    
    logger.info("Starting lambda_handler")

    # Set environment variable for Transformers cache
    cache_dir = '/tmp/transformers_cache'
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Define model paths based on window size
    bucket = 'stock-news-sentiment'
    model_key = 'models/best_model/best_model.pth'
    
    # Load the model and tokenizer
    s3 = boto3.client('s3')
    model_path = '/tmp/model.pth'
    logger.info(f"Downloading model from S3 bucket {bucket} with key {model_key}")
    s3.download_file(bucket, model_key, model_path)
    
    logger.info("Model downloaded, loading model")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(model_path)

    # Parse input
    input_data = event['data']
    
    # Preprocess the input data
    logger.info("Preprocessing input data")
    preprocessed_data = preprocess_input(input_data)
    headlines = [item['preprocessed_headline'] for item in preprocessed_data]
    
    logger.info(f"Preprocessed input data: {headlines}")
    # Create a DataLoader for inference
    max_len = 128
    batch_size = 16
    data_loader = create_data_loader(headlines, tokenizer, max_len, batch_size)
    
    logger.info("Performing inference")
    predictions = []
    top_words_weights = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]

            # Get logits and attention weights from the model
            logits, attentions = model(input_ids=input_ids, attention_mask=attention_mask)
        
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds)

            # Process attention weights for each input
            for i in range(len(input_ids)):
                attention_weights = attentions[-1][i].mean(0)  # Use the last layer's attention and average over heads
                top_words = get_top_n_words(input_ids[i], attention_weights, tokenizer, n=10)
                top_words_weights.append(top_words)
    
    # Prepare the output
    logger.info("Preparing output")
    output = []
    for i, entry in enumerate(preprocessed_data):
        output.append({
            'date': entry['date'],
            'headline': entry['preprocessed_headline'],
            'prediction': "positive" if predictions[i].item() == 1 else "negative",
            'top_10_words': top_words_weights[i]  # Include top words with attention weights
        })
    
    return {
        'statusCode': 200,
        'body': output
    }
