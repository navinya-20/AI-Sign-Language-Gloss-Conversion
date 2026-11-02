from gtts import gTTS
import os
import uuid


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AUDIO_FOLDER = os.path.join(
    BASE_DIR,
    "audio"
)

os.makedirs(
    AUDIO_FOLDER,
    exist_ok=True
)


LANG_CODES = {
    "english": "en",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "hindi": "hi"
}


def generate_speech(text, language):

    language = language.lower()

    if language not in LANG_CODES:
        return None

    if not text or text.strip() == "":
        return None

    filename = str(uuid.uuid4()) + ".mp3"

    filepath = os.path.join(
        AUDIO_FOLDER,
        filename
    )

    language_code = LANG_CODES[language]

    tts = gTTS(
        text=text,
        lang=language_code
    )

    tts.save(filepath)

    return filepath