import torch
import torch.nn as nn
import numpy as np
import json
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

# -----------------------------
# Settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best.pt"
LABEL_PATH = "label2id_subset.json"
VIDEO_PATH = "data/99398745659301-ADMIT.npy"

print("Device:", DEVICE)

# -----------------------------
# Load label mapping
# -----------------------------
with open(LABEL_PATH, "r") as f:
    label2id = json.load(f)

# IMPORTANT:
# Kaggle created the model labels using sorted gloss names
classes = sorted(label2id.keys())

num_classes = len(classes)

print("Number of classes:", num_classes)

# -----------------------------
# Create Swin3D-T
# -----------------------------
weights = Swin3D_T_Weights.KINETICS400_V1

model = swin3d_t(weights=weights)

model.head = nn.Linear(
    model.head.in_features,
    num_classes
)

# -----------------------------
# Load trained model
# -----------------------------
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state"])

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Checkpoint validation accuracy:",
      checkpoint.get("val_acc", "unknown"))

# -----------------------------
# Load preprocessed .npy
# -----------------------------
arr = np.load(VIDEO_PATH).astype(np.float32)

print("Original shape:", arr.shape)

# Same preprocessing used during testing
arr = arr / 255.0

# (T, H, W, C)
#        ↓
# (C, T, H, W)
frames = torch.from_numpy(arr).float()
frames = frames.permute(3, 0, 1, 2)

# Add batch dimension
frames = frames.unsqueeze(0)

frames = frames.to(DEVICE)

print("Model input shape:", frames.shape)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    outputs = model(frames)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(
        probabilities, dim=1
    )

predicted_index = prediction.item()
confidence = confidence.item()

predicted_label = classes[predicted_index]

# -----------------------------
# Result
# -----------------------------
print("\n==============================")
print("       PREDICTION")
print("==============================")
print("Predicted sign :", predicted_label)
print("Confidence     :", f"{confidence * 100:.2f}%")
print("Class index    :", predicted_index)
print("==============================")