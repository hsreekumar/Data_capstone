import subprocess
import sys
import os
import tarfile

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('torch')
install('transformers')
install('seaborn')
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, RocCurveDisplay, auc
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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


# Extract the model from the tar.gz file
model_tar_path = '/opt/ml/processing/model/model.tar.gz'
extract_path = '/opt/ml/processing/model'
with tarfile.open(model_tar_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)

# Assuming the extracted model file is named 'model.pth'
model_file_path = os.path.join(extract_path, 'best_model.pth')

model = NewsClassifier(n_classes=2)
# Load the state dictionary
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))


model.eval()


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load test data
test_data_path = '/opt/ml/processing/test/test.csv'
test_data = pd.read_csv(test_data_path)

test_headlines = test_data['headline'].tolist()
y_test = test_data['label'].tolist()

# Create DataLoader
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
            truncation=True,  # Explicitly specify truncation
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

def create_data_loader(headlines, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(
        headlines=headlines,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

BATCH_SIZE = 16
MAX_LEN = 128

test_data_loader = create_data_loader(test_headlines, y_test, tokenizer, MAX_LEN, BATCH_SIZE)

# Evaluate the model
y_test_pred = []
y_test_true = []

with torch.no_grad():
    for d in test_data_loader:
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        labels = d["labels"]
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        
        y_test_pred.extend(preds)
        y_test_true.extend(labels)

y_test_pred = torch.stack(y_test_pred).cpu()
y_test_true = torch.stack(y_test_true).cpu()



# Existing evaluation logic...

# Compute and print confusion matrix and classification report
cm = confusion_matrix(y_test_true, y_test_pred)
classification_rep = classification_report(y_test_true, y_test_pred, output_dict=True)

# Save confusion matrix plot
try:
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('/opt/ml/processing/evaluation/confusion_matrix.png')
    logging.info('Confusion matrix plot saved successfully.')
except Exception as e:
    logging.error(f'Error saving confusion matrix plot: {e}')

# Compute AUC score
try:
    fpr, tpr, _ = roc_curve(y_test_true, y_test_pred)
    auc_score = auc(fpr, tpr)

    report_dict = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }

    # Save AUC score as JSON
    with open('/opt/ml/processing/evaluation/evaluation.json', 'w') as f:
        json.dump(report_dict, f)
    logging.info('AUC score saved successfully.')
except Exception as e:
    logging.error(f'Error computing or saving AUC score: {e}')

# Save classification report as JSON
try:
    with open('/opt/ml/processing/evaluation/classification_report.json', 'w') as f:
        json.dump(classification_rep, f)
    logging.info('Classification report saved successfully.')
except Exception as e:
    logging.error(f'Error saving classification report: {e}')