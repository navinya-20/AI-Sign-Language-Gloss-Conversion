
import os
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t


# ============================================================
# PATHS
# Keep best.pt and label2id_subset.json in the same folder
# as this video_predict.py file.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LABEL_PATH = os.path.join(BASE_DIR, "label2id_subset.json")

NUM_FRAMES = 16
FRAME_SIZE = 112
TOP_K = 5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD LABELS
# IMPORTANT:
# This alphabetical class order must match the class order
# used when training the model.
# ============================================================

if not os.path.isfile(LABEL_PATH):
    raise FileNotFoundError(
        f"Label file not found: {LABEL_PATH}"
    )

with open(LABEL_PATH, "r", encoding="utf-8") as file:
    label2originalid = json.load(file)

if not isinstance(label2originalid, dict):
    raise ValueError(
        "label2id_subset.json must contain a JSON dictionary."
    )

classes = sorted(label2originalid.keys())

if not classes:
    raise ValueError("No class labels were found in the label file.")

num_classes = len(classes)

print("Number of classes:", num_classes)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if not isinstance(checkpoint, dict):
    raise ValueError(
        "Unexpected checkpoint format. Expected a dictionary "
        "containing the 'model_state' key."
    )

if "model_state" not in checkpoint:
    raise KeyError(
        "The checkpoint does not contain 'model_state'. "
        "Check that best.pt is the correct checkpoint."
    )

model = swin3d_t(weights=None)

model.head = nn.Linear(
    model.head.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state"],
    strict=True
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
print("Checkpoint:", MODEL_PATH)


# ============================================================
# PREPROCESS ONE FRAME
# Keep this consistent with the training preprocessing:
# BGR -> RGB, resize to 112 x 112, scale pixels to [0, 1].
# ============================================================

def preprocess_frame(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Received an empty video frame.")

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = cv2.resize(
        frame,
        (FRAME_SIZE, FRAME_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    frame = frame.astype(np.float32) / 255.0

    return frame


# ============================================================
# READ VIDEO AND SELECT 16 FRAMES
# ============================================================

def extract_video_frames(video_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file does not exist: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    all_frames = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            processed_frame = preprocess_frame(frame)
            all_frames.append(processed_frame)

    finally:
        cap.release()

    if len(all_frames) == 0:
        raise ValueError(
            "No frames could be decoded from the video."
        )

    print("Decoded frames:", len(all_frames))

    # Select 16 evenly spaced frames if possible.
    if len(all_frames) >= NUM_FRAMES:
        indices = np.linspace(
            0,
            len(all_frames) - 1,
            NUM_FRAMES
        ).astype(np.int64)

        selected_frames = [
            all_frames[index] for index in indices
        ]

    else:
        # For short videos, retain every frame and repeat
        # the last frame until there are 16 frames.
        selected_frames = list(all_frames)

        while len(selected_frames) < NUM_FRAMES:
            selected_frames.append(
                selected_frames[-1].copy()
            )

    print("Selected frames:", len(selected_frames))

    return np.stack(selected_frames, axis=0)


# ============================================================
# PREDICT VIDEO
# Return format remains:
# (predicted_gloss, confidence_percentage)
# On failure, return (None, 0).
# ============================================================

def predict_video(video_path):
    print("\n" + "=" * 45)
    print("Processing video:", video_path)
    print("=" * 45)

    try:
        # Shape: (T, H, W, C)
        frames_array = extract_video_frames(video_path)

        # Convert to float tensor.
        frames_tensor = torch.from_numpy(
            frames_array
        ).float()

        # (T, H, W, C) -> (C, T, H, W)
        frames_tensor = frames_tensor.permute(
            3, 0, 1, 2
        )

        # Add batch dimension:
        # (C, T, H, W) -> (1, C, T, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)

        frames_tensor = frames_tensor.to(device)

        print("Model input shape:", tuple(frames_tensor.shape))

        # Run inference.
        with torch.inference_mode():
            logits = model(frames_tensor)

            if logits.ndim != 2:
                raise ValueError(
                    f"Unexpected model output shape: {tuple(logits.shape)}"
                )

            if logits.shape[1] != num_classes:
                raise ValueError(
                    "Model output count does not match the number "
                    "of labels. Check the checkpoint and label file."
                )

            probabilities = torch.softmax(logits, dim=1)[0]

            k = min(TOP_K, num_classes)

            top_probs, top_indices = torch.topk(
                probabilities,
                k=k
            )

        # Display top predictions.
        print("\nTop predictions:")

        for rank, (prob, idx) in enumerate(
            zip(top_probs, top_indices),
            start=1
        ):
            gloss = classes[idx.item()]
            confidence_value = prob.item() * 100

            print(
                f"{rank}. {gloss}: "
                f"{confidence_value:.2f}%"
            )

        predicted_index = top_indices[0].item()
        predicted_gloss = classes[predicted_index]
        confidence = top_probs[0].item() * 100

        print("\nPredicted gloss:", predicted_gloss)
        print(f"Confidence: {confidence:.2f}%")
        print("=" * 45)

        return predicted_gloss, confidence

    except Exception as error:
        print("\nERROR during video prediction:")
        print(f"{type(error).__name__}: {error}")
        print("=" * 45)

        return None, 0.0


# ============================================================
# OPTIONAL TEST MODE
# Runs only when you execute this file directly.
# The Flask app can import predict_video() without running
# this test section.
# ============================================================

if __name__ == "__main__":
    VIDEO_PATH = input(
        "Enter the full path of a test video: "
    ).strip().strip('"')

    predicted_gloss, confidence = predict_video(VIDEO_PATH)

    if predicted_gloss is not None:
        print("\nRecognition result")
        print("Gloss:", predicted_gloss)
        print(f"Confidence: {confidence:.2f}%")

        # Optional translation test.
        try:
            from translation import translate_all_languages

            result = translate_all_languages(predicted_gloss)

            print("\nTranslations:")
            for language, text in result.items():
                print(f"{language}: {text}")

        except ImportError:
            print(
                "\nTranslation module was not loaded. "
                "The gloss prediction was successful."
            )
        except Exception as error:
            print(
                "\nGloss prediction succeeded, but translation failed:"
            )
            print(f"{type(error).__name__}: {error}")
