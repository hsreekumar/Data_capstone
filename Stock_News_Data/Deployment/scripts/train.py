import subprocess
import sys
import os

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('transformers')

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging
import boto3

# Suppress specific warning messages
warnings.filterwarnings("ignore")


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, headlines, labels, tokenizer, max_len):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, item):
        headline = str(self.headlines[item])
        label = self.labels[item]

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
            'headline_text': headline,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(headlines, labels, tokenizer, max_len, batch_size, sampler):
    ds = NewsDataset(
        headlines=headlines,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler(ds),
        num_workers=4
    )

class NewsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.batch_norm1 = nn.GroupNorm(32, 256)  # 32 groups can be adjusted
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

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def train(args):

    # Constants
    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 4
    PATIENCE = 5
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed data
    train_data_path = '/opt/ml/input/data/train/train.csv'
    validation_data_path = '/opt/ml/input/data/validation/validation.csv'
    test_data_path = '/opt/ml/input/data/test/test.csv'

    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)
    test_data = pd.read_csv(test_data_path)

    logging.info(f'\nTrain Data:\n{train_data.head().to_string()}')
    logging.info(f'\nTest Data:\n{test_data.head().to_string()}')
    logging.info(f'\nValidation Data:\n{validation_data.head().to_string()}')

    train_headlines = train_data['headline'].tolist()
    y_train = train_data['label'].tolist()

    validation_headlines = validation_data['headline'].tolist()
    y_validation = validation_data['label'].tolist()

    test_headlines = test_data['headline'].tolist()
    y_test = test_data['label'].tolist()

    # DataLoaders with appropriate samplers
    train_data_loader = create_data_loader(train_headlines, y_train, tokenizer, MAX_LEN, args['batch_size'], RandomSampler)
    validation_data_loader = create_data_loader(validation_headlines, y_validation, tokenizer, MAX_LEN, args['batch_size'], SequentialSampler)
    test_data_loader = create_data_loader(test_headlines, y_test, tokenizer, MAX_LEN, args['batch_size'], SequentialSampler)

    # Model
    model = NewsClassifier(n_classes=2)
    model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    total_steps = len(train_data_loader) * args['epochs']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    best_loss = float('inf')
    best_accuracy = 0
    best_model = None
    patience_counter = 0

    for epoch in range(args['epochs']):
        print(f"Epoch {epoch + 1}/{args['epochs']}")
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_headlines)
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
            model,
            validation_data_loader,
            loss_fn,
            device,
            len(validation_headlines)
        )
        
        print(f'Val loss {val_loss}')
        print(f'Val accuracy {val_acc}')
        
        if val_acc > best_accuracy:
            best_loss = val_loss
            best_accuracy = val_acc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    # Load the best model
    model.load_state_dict(best_model)

    # Evaluation on test set
    y_test_pred = []
    y_test_true = []

    with torch.no_grad():
        for d in test_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            
            y_test_pred.extend(preds)
            y_test_true.extend(labels)

    y_test_pred = torch.stack(y_test_pred).cpu()
    y_test_true = torch.stack(y_test_true).cpu()

    print("\nConfusion Matrix (test):")
    cm = confusion_matrix(y_test_true, y_test_pred)
    print(cm)

    print("\nClassification Report (test):")
    print(classification_report(y_test_true, y_test_pred))

    save_cm = True  # Set to False if you want to display instead of saving

    if save_cm:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('/opt/ml/model/confusion_matrix.png')
    else:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    # Save the model
    model_save_path = "/opt/ml/model/best_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # Upload model to S3
    s3 = boto3.client('s3')
    bucket_name = args['bucket_name']
    s3.upload_file(model_save_path, bucket_name, 'models/news_classifier/best_model.pth')
    print(f"Model uploaded to S3 bucket: {bucket_name}/models/news_classifier/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--bucket_name', type=str, default='your-s3-bucket-name', help='S3 bucket name for saving the model')

    args = vars(parser.parse_args())
    

    train(args)
