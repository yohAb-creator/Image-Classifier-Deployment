
# Use an official, lightweight Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# The --no-cache-dir option keeps the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for the model and copy your model weights into it
COPY cifar10_resnet18_feature_extractor.pt  ./model/

# Copy your application code into the container at /app
COPY app.py .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define an environment variable for the model path inside the container
ENV MODEL_PATH=/app/model/cifar10_resnet18_feature_extractor.pt 

# Run the app using gunicorn when the container launches
# Gunicorn is a robust production web server for Python
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
