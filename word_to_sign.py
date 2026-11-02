import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WORD_VIDEO_FILE = os.path.join(
    BASE_DIR,
    "word_to_videos_local.pkl"
)

REVERSE_FILE = os.path.join(
    BASE_DIR,
    "reverse_lookup_final.pkl"
)


# Load word -> videos
with open(WORD_VIDEO_FILE, "rb") as f:
    word_to_videos = pickle.load(f)


# Load translations -> gloss
with open(REVERSE_FILE, "rb") as f:
    reverse_data = pickle.load(f)


reverse_lookup = reverse_data["reverse_lookup"]


print("Word-to-sign files loaded successfully!")
print("Number of glosses:", len(word_to_videos))


def find_gloss(word, language="english"):

    word = word.strip()

    if not word:
        return None


    # -------------------------
    # ENGLISH
    # -------------------------

    if language.lower() == "english":

        # Direct gloss match
        upper_word = word.upper()

        if upper_word in word_to_videos:
            return upper_word

        # Try normal case-insensitive match
        for gloss in word_to_videos:

            if gloss.lower() == word.lower():
                return gloss

        return None


    # -------------------------
    # INDIAN LANGUAGES
    # -------------------------

    language = language.lower()

    if language not in reverse_lookup:
        return None


    language_dictionary = reverse_lookup[language]


    # Exact match
    if word in language_dictionary:
        return language_dictionary[word]


    # Case-insensitive match
    for translated_word, gloss in language_dictionary.items():

        if translated_word.lower() == word.lower():
            return gloss


    return None


def get_videos_for_gloss(gloss):

    if not gloss:
        return []


    videos = word_to_videos.get(gloss, [])

    existing_videos = []


    for video_path in videos:

        if os.path.exists(video_path):
            existing_videos.append(video_path)


    return existing_videos