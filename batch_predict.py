import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os

from torchvision.models.video import swin3d_t
from translation import translate_all_languages


# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LABEL_PATH = os.path.join(BASE_DIR, "label2id_subset.json")

VIDEO_FOLDER = r"C:\sign_videos"


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==============================
# LOAD LABELS
# ==============================

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label2originalid = json.load(f)

classes = sorted(label2originalid.keys())
num_classes = len(classes)

print("Number of classes:", num_classes)


# ==============================
# LOAD MODEL
# ==============================

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


# ==============================
# VIDEO PREDICTION FUNCTION
# ==============================

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, 0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None, 0

    num_frames = 16

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    frames = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (112, 112)
        )

        frame = frame.astype(
            np.float32
        ) / 255.0

        frames.append(frame)

    cap.release()

    if len(frames) != 16:
        return None, 0

    arr = np.stack(frames)

    tensor = torch.from_numpy(arr).float()

    tensor = tensor.permute(
        3, 0, 1, 2
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # ==============================
    # MODEL PREDICTION
    # ==============================

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()

    confidence = confidence.item() * 100

    predicted_gloss = classes[predicted_index]

    return predicted_gloss, confidence


# ==============================
# FIND VIDEOS
# ==============================

videos = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(".mp4")
]


print()
print("Videos found:", len(videos))
print()


# ==============================
# TEST FIRST 10 VIDEOS
# ==============================

test_videos = videos[:10]

for video in test_videos:

    video_path = os.path.join(
        VIDEO_FOLDER,
        video
    )

    predicted_gloss, confidence = predict_video(
        video_path
    )

    print("----------------------------------")
    print("Video:", video)

    if predicted_gloss is None:

        print("Prediction failed")
        continue

    result = translate_all_languages(
        predicted_gloss
    )

    print("Predicted:", predicted_gloss)
    print("Confidence:", f"{confidence:.2f}%")

    print("Tamil     :", result["Tamil"])
    print("Telugu    :", result["Telugu"])
    print("Kannada   :", result["Kannada"])
    print("Malayalam :", result["Malayalam"])
    print("Hindi     :", result["Hindi"])

print()
print("==================================")
print("       TESTING COMPLETED")
print("==================================")