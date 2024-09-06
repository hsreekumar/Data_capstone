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
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)  # 32 groups can be adjusted
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.pooler_output
        x = self.dropout(cls_output)
        x = self.fc1(x)
        # Log the shape of x before passing to BatchNorm
        logging.info(f'Input tensor shape before BatchNorm: {x.shape}\n')
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

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
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
    
    # Prepare the output
    logger.info("Preparing output")
    output = []
    for i, entry in enumerate(preprocessed_data):
        output.append({
            'date': entry['date'],
            'headline': entry['preprocessed_headline'],
            'prediction': int(predictions[i].item())
        })
    
    return {
        'statusCode': 200,
        'body': output
    }
