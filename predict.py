import torch
import torch.nn as nn
import numpy as np
import json
import os

from torchvision.models.video import swin3d_t
from translation import translate_all_languages


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LABEL_PATH = os.path.join(BASE_DIR, "label2id_subset.json")

# Change this to your test .npy file
NPY_PATH = os.path.join(BASE_DIR, "99398745659301-ADMIT.npy")


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ==========================================
# LOAD LABELS
# ==========================================

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label2originalid = json.load(f)

# IMPORTANT:
# Model output index is based on sorted gloss names
classes = sorted(label2originalid.keys())

num_classes = len(classes)

print("Number of classes:", num_classes)


# ==========================================
# LOAD MODEL
# ==========================================

model = swin3d_t(weights=None)

model.head = nn.Linear(
    model.head.in_features,
    num_classes
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state"])

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ==========================================
# LOAD NPY
# ==========================================

arr = np.load(NPY_PATH).astype(np.float32) / 255.0

print("Original shape:", arr.shape)


# ==========================================
# CONVERT TO MODEL FORMAT
# ==========================================

frames = torch.from_numpy(arr).float()

# (T,H,W,C) → (C,T,H,W)
frames = frames.permute(3, 0, 1, 2)

# Add batch dimension
frames = frames.unsqueeze(0)

frames = frames.to(device)

print("Model input shape:", frames.shape)


# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(frames)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

predicted_index = predicted_index.item()
confidence = confidence.item() * 100

predicted_gloss = classes[predicted_index]


# ==========================================
# TRANSLATION
# ==========================================

result = translate_all_languages(predicted_gloss)


# ==========================================
# FINAL OUTPUT
# ==========================================

print()
print("==============================")
print("       SIGN RECOGNITION")
print("==============================")

print(f"English    : {result['English']}")
print(f"Tamil      : {result['Tamil']}")
print(f"Telugu     : {result['Telugu']}")
print(f"Kannada    : {result['Kannada']}")
print(f"Malayalam  : {result['Malayalam']}")
print(f"Hindi      : {result['Hindi']}")

print()
print(f"Confidence : {confidence:.2f}%")
print(f"Class Index: {predicted_index}")

print("==============================")