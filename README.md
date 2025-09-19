# End-to-End Image Classification API

> MLOps Project: A Dockerized PyTorch image classifier with a Flask REST API for end-to-end deployment.

This repository contains the source code for a complete, end-to-end machine learning system that serves an image classification model through a REST API. The entire application is containerized with Docker for reproducible deployment.

---

## ✨ Key Features
* **RESTful API:** Exposes a `/predict` endpoint to serve image classification predictions.
* **Transfer Learning:** Utilizes a pre-trained ResNet-18 model fine-tuned on the CIFAR-10 dataset for efficient and accurate classification.
* **Containerized & Reproducible:** Fully containerized with Docker, ensuring a consistent environment for development and deployment.
* **End-to-End MLOps:** Covers the complete lifecycle from model training in a Jupyter Notebook to a production-ready API.

---

## 🛠️ Technologies Used
* **Model Development:** Python, PyTorch, NumPy
* **API Backend:** Flask
* **MLOps & Deployment:** Docker, AWS (designed for)
* **Version Control:** Git & GitHub

---

## 🏗️ Project Architecture
This system consists of four primary components:

* **The Model:** A ResNet-18 convolutional neural network trained on the CIFAR-10 dataset using transfer learning in PyTorch. The complete training and analysis process is documented in the `ML_API.ipynb` notebook.
* **The API:** A web service built with Flask (`app.py`) that exposes a `/predict` endpoint. It receives an image, preprocesses it, and returns a JSON object with the predicted class and a confidence score.
* **The Container:** A Docker image containing the Flask application, the trained model, and all necessary dependencies. This ensures a reproducible and portable deployment environment. The blueprint for this container is the `Dockerfile`.
* **The Deployment:** The final Docker container is designed to be deployed on a cloud service like AWS EC2, making the API publicly accessible.

---

## 🚀 Getting Started

### Prerequisites
* [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine.
* [Git](https://git-scm.com/) for cloning the repository.

### 1. Clone the Repository
```bash
git clone [https://github.com/yohAb-creator/Image-Classifier-Deployment.git](https://github.com/yohAb-creator/Image-Classifier-Deployment.git)
cd Image-Classifier-Deployment

2. Build the Docker Image

From the root of the project directory, execute the following command:
Bash

docker build -t image-classifier-app .

3. Run the Docker Container

Bash

docker run -p 5000:80 image-classifier-app

The API will now be running and accessible on your local machine.

4. Make a Prediction

The API exposes a POST endpoint at /predict. You can send an image file to this endpoint to get a classification result. Use a tool like curl or Postman.

Here is an example using curl from your terminal:
Bash

# Note: Replace '/path/to/your/image.jpg' with the actual path to an image file.
curl -X POST -F "file=@/path/to/your/image.jpg" [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)

Expected Response

The API will return a JSON object containing the predicted class label and the model's confidence score.
JSON

{
  "class_name": "dog",
  "confidence": 0.9251
}

📁 Project Structure

For maximum clarity and ease of use, all essential files are located in the root of this repository:

.
├── Image_Classifier.ipynb
├── app.py
├── cifar10_resnet18_feature_extractor.pt
├── Dockerfile
└── requirements.txt

🔮 Future Improvements

    CI/CD Pipeline: Implement GitHub Actions to automatically build and test the Docker container on push.

    Cloud Deployment Scripts: Add Terraform or AWS CDK scripts for automated infrastructure provisioning on AWS EC2 or ECS.

    Batch Predictions: Create a /predict_batch endpoint to handle multiple images in a single request.

    Interactive Frontend: Develop a simple Streamlit or React frontend to allow users to upload images and see results in a browser.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

📫 Contact

Yohannes Abateneh - GitHub Profile - LinkedIn Profile

