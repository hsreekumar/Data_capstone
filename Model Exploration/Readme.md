## **<u>Exploring Different Models</u>**

From the exploratory analysis of Models, we find that traditional models like RadomForestClassifier & Naive Bayes offer great accuracies, around 80% for sentiment prediction of stock movement. More advanced models like LSTM, GRU etc doesn't seem to suit the purpose and provide lower accuracies, hovering around 70%.

Further experimenting with more advanced HuggingFace models for NLP like BERT seems to have promising results. The FinBERT model designed for stock sentiment prediction doesn't seem to suit the purpose, out of the box, and has only less than 50% accuracy. While the BertForSequenceClassification model came back with 84% accuracy. Further tuning the Base BERT model with a custom classification head seemed to improve the accuracy to 85%.

The BERT model has shown promising results, indicating that it’s well-aligned with the objectives of the task. To further enhance its performance, the next steps will involve fine-tuning the hyperparameters and making adjustments to the model’s layers. These refinements are expected to improve the model’s accuracy and overall effectiveness, bringing us closer to optimal outcomes.
