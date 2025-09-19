\# End-to-End Image Classification API



This repository contains the source code for a complete, end-to-end machine learning system that serves an image classification model through a REST API. The entire application is containerized with Docker for reproducible deployment.



\*\*Project Structure:\*\* For maximum clarity and ease of use, all essential files (`ML\_API.ipynb`, `app.py`, `Dockerfile`, `requirements.txt`, and the model artifact) are located in the root of this repository.



\## Project Architecture



This system consists of four primary components:



1\.  \*\*The Model:\*\* A ResNet-18 convolutional neural network trained on the CIFAR-10 dataset using transfer learning in PyTorch. The complete training and analysis process is documented in the `ML\_API.ipynb` notebook.



2\.  \*\*The API:\*\* A web service built with Flask (`app.py`) that exposes a `/predict` endpoint. It receives an image, preprocesses it, and returns a JSON object with the predicted class and a confidence score.



3\.  \*\*The Container:\*\* A Docker image containing the Flask application, the trained model, and all necessary dependencies. This ensures a reproducible and portable deployment environment. The blueprint for this container is the `Dockerfile`.



4\.  \*\*The Deployment:\*\* The final Docker container is designed to be deployed on a cloud service like AWS EC2, making the API publicly accessible.



\## How to Run Locally



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/YourUsername/your-repository-name.git](https://github.com/YourUsername/your-repository-name.git)

&nbsp;   cd your-repository-name

&nbsp;   ```



2\.  \*\*Train the model (Optional):\*\*

&nbsp;   To retrain the model or inspect the methodology, open the `ML\_API.ipynb` notebook. Running all cells will regenerate the `cifar10\_resnet18\_feature\_extractor.pt` file in this directory.



3\.  \*\*Build the Docker image:\*\*

&nbsp;   From the root of the project directory, execute the following command:

&nbsp;   ```bash

&nbsp;   docker build -t image-classifier-app .

&nbsp;   ```



4\.  \*\*Run the Docker container:\*\*

&nbsp;   ```bash

&nbsp;   docker run -p 5000:80 image-classifier-app

&nbsp;   ```



The API will now be accessible at `http://127.0.0.1:5000/predict`.

