from flask import Flask, request, jsonify, render_template, send_file

from word_to_sign import find_gloss, get_videos_for_gloss
from tts import generate_speech
from video_predict import predict_video

import os
import uuid


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return jsonify({
            "error": "No video uploaded"
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "error": "No video selected"
        }), 400

    extension = os.path.splitext(video.filename)[1].lower()

    if not extension:
        extension = ".webm"

    filename = str(uuid.uuid4()) + extension

    video_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    video.save(video_path)

    try:

        predicted_gloss, confidence = predict_video(
            video_path
        )

        if predicted_gloss is None:
            return jsonify({
                "error": "Could not process video"
            }), 500

        from translation import translate_all_languages

        result = translate_all_languages(
            predicted_gloss
        )

        response = {
            "english": result["English"],
            "tamil": result["Tamil"],
            "telugu": result["Telugu"],
            "kannada": result["Kannada"],
            "malayalam": result["Malayalam"],
            "hindi": result["Hindi"],
            "confidence": round(confidence, 2)
        }

        return jsonify(response)

    except Exception as e:

        print("PREDICTION ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(video_path):

            try:
                os.remove(video_path)

            except Exception:
                pass


@app.route("/word-to-sign", methods=["POST"])
def word_to_sign():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No data received"
        }), 400

    word = data.get(
        "word",
        ""
    ).strip()

    language = data.get(
        "language",
        "english"
    ).strip().lower()

    if not word:
        return jsonify({
            "error": "Please enter a word"
        }), 400

    gloss = find_gloss(
        word,
        language
    )

    if gloss is None:
        return jsonify({
            "error": f"No sign found for '{word}'"
        }), 404

    videos = get_videos_for_gloss(
        gloss
    )

    if not videos:
        return jsonify({
            "error": f"No video available for '{gloss}'"
        }), 404

    video_urls = []

    for video_path in videos:

        filename = os.path.basename(
            video_path
        )

        video_urls.append(
            "/sign-video/" + filename
        )

    return jsonify({
        "word": word,
        "language": language,
        "gloss": gloss,
        "videos": video_urls,
        "count": len(video_urls)
    })


@app.route("/sign-video/<filename>")
def sign_video(filename):

    video_folder = r"C:\sign_videos"

    video_path = os.path.abspath(
        os.path.join(
            video_folder,
            filename
        )
    )

    video_folder = os.path.abspath(
        video_folder
    )

    if not video_path.startswith(
        video_folder + os.sep
    ):
        return "Invalid video path", 403

    if not os.path.exists(video_path):
        return "Video not found", 404

    return send_file(video_path)


@app.route("/tts", methods=["POST"])
def text_to_speech():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No data received"
        }), 400

    text = data.get(
        "text",
        ""
    ).strip()

    language = data.get(
        "language",
        ""
    ).strip().lower()

    if not text:
        return jsonify({
            "error": "No text provided"
        }), 400

    if not language:
        return jsonify({
            "error": "No language provided"
        }), 400

    try:

        audio_path = generate_speech(
            text,
            language
        )

        if audio_path is None:
            return jsonify({
                "error": "Unsupported language"
            }), 400

        filename = os.path.basename(
            audio_path
        )

        return jsonify({
            "audio": "/audio/" + filename
        })

    except Exception as e:

        print("TTS ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/audio/<filename>")
def serve_audio(filename):

    audio_folder = os.path.abspath(
        os.path.join(
            BASE_DIR,
            "audio"
        )
    )

    audio_path = os.path.abspath(
        os.path.join(
            audio_folder,
            filename
        )
    )

    if not audio_path.startswith(
        audio_folder + os.sep
    ):
        return "Invalid audio path", 403

    if not os.path.exists(audio_path):
        return "Audio not found", 404

    return send_file(
        audio_path,
        mimetype="audio/mpeg"
    )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )