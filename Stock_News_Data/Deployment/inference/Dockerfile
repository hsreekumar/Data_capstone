# syntax=docker/dockerfile:1
FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
RUN pip install --no-cache-dir torch torchvision pandas transformers scikit-learn boto3 nltk

# Download NLTK data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords wordnet

# Set environment variable for Transformers cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

# Copy function code into the container
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your Lambda handler function
CMD ["lambda_function.lambda_handler"]

