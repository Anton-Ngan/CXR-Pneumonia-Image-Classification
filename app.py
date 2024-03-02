from pathlib import Path
import gradio as gr
import torch  
import torchvision.transforms as transforms
from pneumonia_model import PneumoniaModel

# Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
IMG_SIZE = 224
CLASS_LABELS = ["Normal", "Pneumonia"]

# Load the model and its params
model = PneumoniaModel(input_shape=3,
                       hidden_units=16,
                       output_shape=1
                        ).to(device)
checkpoint = torch.load(Path("model_state_dict_results/pneumonia_cxr_model.pth"), map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Example images
examples = []
n = 2
p = 2
for i in range(1, n+1):
    examples.append("examples/n" + str(i) + ".jpeg")
for i in range(1, p+1):
    examples.append("examples/p" + str(i) + ".jpeg")

def process_image(image):
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    image_tensor = transformation(image).unsqueeze(0)
    
    return image_tensor

def classify_image(image):
    model.eval()
    with torch.inference_mode():
        img_tensor = process_image(image)
        y_logit = model(img_tensor.to(device))
        y_prob = torch.sigmoid(y_logit)
        y_pred = torch.round(y_prob)

        predicted_class = CLASS_LABELS[int(y_pred.item())]
        probability = y_prob if (y_pred == 1) else 1-y_prob

        return {predicted_class: probability,
                CLASS_LABELS[1-int(y_pred.item())]: 1 - probability}

# Deploy the model
gr.Interface(fn=classify_image, 
             inputs=gr.Image(type="pil"),
             outputs="label",
             examples=examples,
             ).launch(share=True)