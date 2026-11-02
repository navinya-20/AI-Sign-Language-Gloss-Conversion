import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import re
import csv

from torchvision.models.video import swin3d_t


# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LABEL_PATH = os.path.join(BASE_DIR, "label2id_subset.json")

VIDEO_FOLDER = r"C:\sign_videos"

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "evaluation_results.csv"
)


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

model.load_state_dict(
    checkpoint["model_state"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ==============================
# EXTRACT ACTUAL GLOSS
# ==============================

def get_actual_gloss(filename):

    # Remove extension
    name = os.path.splitext(filename)[0]

    # Filename format:
    # 0004970838496565211-GUIDE

    if "-" not in name:
        return name

    gloss = name.split("-", 1)[1]

    # Convert URL encoded spaces
    gloss = gloss.replace("%20", " ")

    return gloss


# ==============================
# NORMALIZE GLOSS
# ==============================

def normalize_gloss(gloss):

    gloss = gloss.upper()

    # Remove spaces
    gloss = gloss.replace(" ", "")

    # Remove special characters
    gloss = re.sub(
        r"[^A-Z0-9]",
        "",
        gloss
    )

    return gloss


# ==============================
# PREDICT VIDEO
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

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        frame = cv2.resize(
            frame,
            (112, 112)
        )

        # Normalize
        frame = (
            frame.astype(np.float32)
            / 255.0
        )

        frames.append(frame)

    cap.release()

    if len(frames) != 16:
        return None, 0

    arr = np.stack(frames)

    tensor = torch.from_numpy(
        arr
    ).float()

    # T,H,W,C -> C,T,H,W
    tensor = tensor.permute(
        3, 0, 1, 2
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # ==============================
    # INFERENCE
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

    predicted_index = (
        predicted_index.item()
    )

    confidence = (
        confidence.item() * 100
    )

    predicted_gloss = classes[
        predicted_index
    ]

    return predicted_gloss, confidence


# ==============================
# FIND VIDEOS
# ==============================

videos = sorted([
    f
    for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(".mp4")
])

print()
print("Videos found:", len(videos))
print()

if len(videos) == 0:

    print("ERROR: No MP4 videos found!")

    exit()


# ==============================
# EVALUATION
# ==============================

results = []

correct = 0
failed = 0

total = len(videos)


for i, video in enumerate(videos, start=1):

    video_path = os.path.join(
        VIDEO_FOLDER,
        video
    )

    actual = get_actual_gloss(video)

    predicted, confidence = predict_video(
        video_path
    )

    if predicted is None:

        failed += 1

        print(
            f"[{i}/{total}] FAILED - {video}"
        )

        results.append({
            "video": video,
            "actual": actual,
            "predicted": "FAILED",
            "confidence": 0,
            "correct": False
        })

        continue


    actual_normalized = normalize_gloss(
        actual
    )

    predicted_normalized = normalize_gloss(
        predicted
    )

    is_correct = (
        actual_normalized
        == predicted_normalized
    )

    if is_correct:
        correct += 1

    status = "CORRECT" if is_correct else "WRONG"

    print(
        f"[{i}/{total}] "
        f"{status} | "
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"{confidence:.2f}%"
    )

    results.append({
        "video": video,
        "actual": actual,
        "predicted": predicted,
        "confidence": round(
            confidence,
            2
        ),
        "correct": is_correct
    })


# ==============================
# ACCURACY
# ==============================

successful = total - failed

if successful > 0:

    accuracy = (
        correct / successful
    ) * 100

else:

    accuracy = 0


# ==============================
# SAVE CSV
# ==============================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video",
            "actual",
            "predicted",
            "confidence",
            "correct"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ==============================
# FINAL RESULT
# ==============================

print()
print("======================================")
print("          EVALUATION COMPLETE")
print("======================================")

print(
    f"Total videos      : {total}"
)

print(
    f"Successful videos : {successful}"
)

print(
    f"Failed videos     : {failed}"
)

print(
    f"Correct           : {correct}"
)

print(
    f"Wrong             : {successful - correct}"
)

print(
    f"Accuracy          : {accuracy:.2f}%"
)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_CSV
)

print("======================================")