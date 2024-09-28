# **<u>Training & Deployment</u>**

### **<u> Model Deployment Architecture</u>**

![](https://github.com/hsreekumar/Data_capstone/blob/main/Deployment/Architecture.png?raw=true)

### **Steps Followed**

- Training is done in AWS Sagemaker, referencing data stored in S3 bucket. Once all the steps in the training pipeline( Pre-processing, tuning, evaluation) are done, the best model is registered in S3

- Inference/prediction is done with reference to the best model saved in S3.

  -  A Lambda function is created to load the model, read the user input (news headlines for sentiment prediction), preprocess the input and predict whether it is positive or negative by evaluating against the trained model

  - Lambda function is compiled into a docker image, bundled with all the desired dependencies and saved in Elastic Container Registry

  - Lambda function is deployed pointing to the latest ECR docker image

  - Api gateway is created to expose a rest endpoint, to read the json input from user and pass it on to the Lambda function for inference

- A streamlit app is created, which inputs a date, and news headlines from users in an interactive UI.

- Another utility app is deployed in streamlit to grab news headlines to provide input to the stock prediction app.

- The app is hosted in streamlit cloud, making a public IP available to the end user

### **Sagemaker Pipeline**
![](https://raw.githubusercontent.com/hsreekumar/Data_capstone/main/Deployment/output/Sagemaker%20Pipeline.png)

## **Model Training Results**

![](https://github.com/hsreekumar/Data_capstone/blob/main/Deployment/output/confusion_matrix.png)

#### **Evaluation Result**
{"classification_metrics": {"auc_score": {"value": 0.8566028225806451}}}
