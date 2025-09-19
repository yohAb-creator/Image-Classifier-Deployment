import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image

# ==============================================================================
# --- 1. APP INITIALIZATION & MODEL LOADING ---
# ==============================================================================

print("Initializing inference server...")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Define Constants and Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- Specify the path to your model and determine device ---
# This method is robust: it reads the path from an environment variable (for Docker)
# and provides a local fallback path (for PyCharm).
MODEL_PATH = os.getenv("MODEL_PATH", "cifar10_resnet18_feature_extractor.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using computational device: {device}")

# --- Load the Model Architecture ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# --- Load the Trained Weights ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file not found at '{MODEL_PATH}'. Please check the path.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded and ready for inference.")

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def transform_image(image_bytes):
    """
    Takes image file bytes, transforms the image, and returns a tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return data_transforms(image).unsqueeze(0)

# ==============================================================================
# --- 3. FLASK API ENDPOINT ---
# ==============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an image file, makes a prediction, and returns the result.
    """
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            tensor = transform_image(image_bytes)
            tensor = tensor.to(device)

            with torch.no_grad():
                outputs = model(tensor)
                _, preds = torch.max(outputs, 1)
                predicted_class = class_names[preds[0]]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4)
            })
        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "invalid request"})



if __name__ == "__main__":
    # Note: Using debug=True is not recommended for production.
    # A production server like Gunicorn would be used instead.
    app.run(debug=True)