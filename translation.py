import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your existing translation file
TRANSLATION_FILE = os.path.join(
    BASE_DIR,
    "gloss_translations_cleaned.pkl"
)

with open(TRANSLATION_FILE, "rb") as f:
    translations = pickle.load(f)

print("Translation file loaded successfully!")


# ==========================================
# MANUAL CORRECTIONS
# ==========================================

TAMIL_CORRECTIONS = {
    "ADMIT": "ஒப்புக்கொள்"
}

HINDI_CORRECTIONS = {
    "ADMIT": "स्वीकार करना"
}


def translate_all_languages(english_gloss):

    gloss = english_gloss.upper()

    tamil = translations["tamil"].get(gloss, english_gloss)
    telugu = translations["telugu"].get(gloss, english_gloss)
    kannada = translations["kannada"].get(gloss, english_gloss)
    malayalam = translations["malayalam"].get(gloss, english_gloss)
    hindi = translations["hindi"].get(gloss, english_gloss)

    # Apply corrections
    if gloss in TAMIL_CORRECTIONS:
        tamil = TAMIL_CORRECTIONS[gloss]

    if gloss in HINDI_CORRECTIONS:
        hindi = HINDI_CORRECTIONS[gloss]

    return {
        "English": english_gloss,
        "Tamil": tamil,
        "Telugu": telugu,
        "Kannada": kannada,
        "Malayalam": malayalam,
        "Hindi": hindi
    }