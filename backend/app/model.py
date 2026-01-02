import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------------
# CONFIG
# ------------------------
CLASS_NAMES = [
    'L_index_touch', 'L_middle_touch', 'L_no_touch',
    'L_pinky_touch', 'L_ring_touch',
    'R_index_touch', 'R_middle_touch', 'R_no_touch',
    'R_pinky_touch', 'R_ring_touch'
]

DEVICE = torch.device("cpu")

# ------------------------
# PATH HANDLING (CLOUD SAFE)
# ------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_PATH = os.path.join(BASE_DIR, "ml", "best_model.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# ------------------------
# MODEL LOADING
# ------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# ------------------------
# IMAGE TRANSFORMS
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ------------------------
# PREDICTION FUNCTION
# ------------------------
def predict_image(image: Image.Image) -> str:
    """
    Takes a PIL image and returns predicted gesture label.
    Cloud-safe, CPU-only inference.
    """
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()

    return CLASS_NAMES[pred_idx]
