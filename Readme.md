# **Stock news sentiment prediction for Dow Jones Industrial Average(DJI)**

App link: https://stock-news.streamlit.app/

Utility to grab news headlines to test the model : https://news-grabber.streamlit.app/

*How to use the app:*

- Select one or multiple dates
- Provide one or more news headlines for each date selected
- Submit to see whether the model predicted 'positive' or 'negative' movement of stock price

## **<u> Model Deployment Architecture</u>**

![](https://github.com/hsreekumar/Data_capstone/blob/main/Deployment/Architecture.png?raw=true)

## <u>**Problem statement**</u>

Reddit news and DJI stock index price changes are available for 9 years. Analyze the news sentiment and the DJI price fluctuation trends to successfully predict the stock movement based on news headlines.

*Data* - https://github.com/hsreekumar/Data_capstone/tree/main/Data

## <u>**Data Exploration**</u>

Data wrangling, text cleaning, basic sentiment analysis and word clouds are explored to identify the patterns in the available data.

https://github.com/hsreekumar/Data_capstone/tree/main/Data%20Exploration

## **<u>Available research</u>**

Various results are available in gaggle, where the dataset was obtained from. Some of the existing results in kaggle are reproduced and the existing research papers on the topic are listed in the following link.

https://github.com/hsreekumar/Data_capstone/tree/main/Existing%20Research%20and%20Reproduce%20Available%20Solutions

## **<u>Exploring Different Models</u>**

From the exploratory analysis of Models, we find that traditional models like RadomForestClassifier & Naive Bayes offer great accuracies, around 80% for sentiment prediction of stock movement. More advanced models like LSTM, GRU etc doesn't seem to suit the purpose and provide lower accuracies, hovering around 70%.

Further experimenting with more advanced HuggingFace models for NLP like BERT seems to have promising results. The FinBERT model designed for stock sentiment prediction doesn't seem to suit the purpose, out of the box, and has only less than 50% accuracy. While the BertForSequenceClassification model came back with 84% accuracy. Further tuning the Base BERT model with a custom classification head seemed to improve the accuracy to 85%.

The BERT model has shown promising results, indicating that it’s well-aligned with the objectives of the task. To further enhance its performance, the next steps will involve fine-tuning the hyperparameters and making adjustments to the model’s layers. These refinements are expected to improve the model’s accuracy and overall effectiveness, bringing us closer to optimal outcomes.

*Please find the model exploration results below*

https://github.com/hsreekumar/Data_capstone/tree/main/Model%20Exploration

## **<u>Scaling Model for large input</u>**

**Batch Processing**: The DataLoader handles mini-batch processing by dividing the dataset into smaller batches (BATCH_SIZE), which helps in efficient memory management and accelerates training.

During training, each batch is loaded sequentially, allowing the model to process smaller chunks of data.

The shuffle=True parameter ensures that the data is shuffled at the start of each epoch, preventing the model from learning the order of the data.

Gradient clipping (nn.utils.clip_grad_norm_) is used to prevent exploding gradients by capping the gradients during backpropagation.

**Data Parallelism**: The model is wrapped with nn.DataParallel to parallelize it across multiple GPUs.

**Memory Management**: Efficient data loading with multiple worker threads. Avoid unnecessary large memory allocations.

The DataLoader is configured with multiple workers (num_workers=4). This allows for efficient data loading in parallel, reducing the CPU bottleneck and making better use of available memory.

Tensors are moved to the GPU (device) only when they are needed for computation. This prevents unnecessary memory usage on the GPU.

*Please find the code for model scaling below*

https://github.com/hsreekumar/Data_capstone/tree/main/Model%20Scaled%20for%20Input

## **<u>Training & Deployment</u>**



- Training is done in AWS Sagemaker, referencing data stored in S3 bucket. Once all the steps in the training pipeline( Pre-processing, tuning, evaluation) are done, the best model is registered in S3

- Inference/prediction is done with reference to the best model saved in S3.

  -  A Lambda function is created to load the model, read the user input (news headlines for sentiment prediction), preprocess the input and predict whether it is positive or negative by evaluating against the trained model

  - Lambda function is compiled into a docker image, bundled with all the desired dependencies and saved in Elastic Container Registry

  - Lambda function is deployed pointing to the latest ECR docker image

  - Api gateway is created to expose a rest endpoint, to read the json input from user and pass it on to the Lambda function for inference

- A streamlit app is created, which inputs a date, and news headlines from users in an interactive UI.

- Another utility app is deployed in streamlit to grab news headlines to provide input to the stock prediction app.

- The app is hosted in streamlit cloud, making a public IP available to the end user



*Please find the link to the model deployment code, which involves Sagemaker pipeline, Preprocessing, Training, Evaluation & Lambda script, along with UI and and the outputs like Evaluation Metrics, in the link below*

https://github.com/hsreekumar/Data_capstone/tree/main/Deployment



## **Cost, Performance & Retraining**

- Cost is incurred mostly during the training phase, where sagemaker is charged per hour based on the instance type and gpu usage
- Once training is done and image saved in s3, storage cost is minimal as the image is small in size
- The cost associated with ECR for saving docker image is also minimal as long as the image is within 5GB
- Lambda and api-gateway offer 1 million calls per month in free tier, essentially totaling the charges associated to zero.
- EC2 instance has 750 free hours a year. As long as the streamlit app is hosted only as per requirement, cost associated with EC2 is minimal as well
- Performance depends on the frequency of inference conducted. Lambda instance needs to be warmed up for a fast response. Once warm, response times fall with a a few seconds
- Re-training might be required when the prediction is not working as expected for latest news or a different source of news. Retraining involves running sagemaker pipeline against a new dataset saved in S3
- As long as the newly trained image is in the same S3 location as before, lambda function can be redeployed, without having to redeploy, api-gateway or streamlit app for the changes to take effect

