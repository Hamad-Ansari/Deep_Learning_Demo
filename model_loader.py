import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import requests

# ImageNet class labels
def get_imagenet_labels():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(url)
    return {int(key): value[1] for key, value in response.json().items()}

def load_model(model_name="ResNet50"):
    """Load pre-trained model and preprocessing function"""
    
    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    # Load model
    # Map known model names to loader callables to avoid duplicated branches
    model_map = {
        "ResNet50": models.resnet50,
        "VGG16": models.vgg16,
    }
    loader = model_map.get(model_name, models.resnet50)
    # For a custom model, load it before calling this function or set loader accordingly.
    model = loader(pretrained=True)
    
    model.eval()
    
    return model, preprocess

def predict(model, input_tensor, top_k=5):
    """Make prediction and return top classes"""
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert to lists
    top_probs = top_probs.numpy()
    top_indices = top_indices.numpy()
    
    # Get class labels
    imagenet_labels = get_imagenet_labels()
    top_classes = [imagenet_labels[idx] for idx in top_indices]
    
    return output, top_classes, top_probs

def preprocess_image(image, preprocess_fn):
    """Preprocess image for model input"""
    return preprocess_fn(image).unsqueeze(0)