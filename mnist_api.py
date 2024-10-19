import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from model import MNISTNet

# Initialize Flask app
app = Flask(__name__)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define argument parser
parser = argparse.ArgumentParser(description="MNIST Flask API")
parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model weights")

# Parse arguments
args = parser.parse_args()
model_path = args.model_path
#model_path = "weights/mnist_net.pth"
# Initialize and load the model
model = MNISTNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data  # Get image data from POST request
    img_pil = Image.open(io.BytesIO(img_binary))  # Convert binary to PIL image

    # Apply transformations to the image
    tensor = transform(img_pil).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        predicted = outputs.max(1)  # Get the index of the highest probability

    # Return the predicted class as JSON
    return jsonify({"prediction": int(predicted[0])})

# Define batch prediction route for multiple images
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Get the image data from the request
    images_binary = request.files.getlist("images[]")  # Expecting a batch of images as form-data

    tensors = []

    # Process each image in the batch
    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream)  # Read the image
        tensor = transform(img_pil)  # Apply transformations
        tensors.append(tensor)

    # Stack tensors to form a batch tensor
    batch_tensor = torch.stack(tensors, dim=0)

    # Make prediction
    with torch.no_grad():
        outputs = model(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    # Return the batch predictions as JSON
    return jsonify({"predictions": predictions.tolist()})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
