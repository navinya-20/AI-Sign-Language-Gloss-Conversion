import json
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_FILE = os.path.join(BASE_DIR, "label2id_subset.json")
OLD_TRANSLATION_FILE = os.path.join(BASE_DIR, "gloss_translations_cleaned.pkl")
NEW_TRANSLATION_FILE = os.path.join(BASE_DIR, "gloss_translations_corrected.pkl")

# Load 304 classes
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    labels = json.load(f)

# Load existing translations
with open(OLD_TRANSLATION_FILE, "rb") as f:
    translations = pickle.load(f)

# --------------------------------------------------
# Correct translations
# Add more corrections here whenever needed.
# --------------------------------------------------

TAMIL_CORRECTIONS = {
    "ADMIT": "ஒப்புக்கொள்",
    "ALL": "அனைத்தும்",
    "AFTER": "பிறகு",
    "ACTION": "செயல்",
    "ABOUT1": "பற்றி",
    "ALARM": "அலாரம்",
    "ANGRY": "கோபம்",
    "APPLE": "ஆப்பிள்",
    "BATH": "குளியல்",
    "BELIEVE1": "நம்பு",
    "BOY": "சிறுவன்",
    "BREAKFAST1": "காலை உணவு",
    "BROTHER": "சகோதரன்",
    "CANCER": "புற்றுநோய்",
    "DANCE": "நடனம்",
    "DARK1": "இருள்",
    "DINNER1": "இரவு உணவு",
    "DOG1": "நாய்",
    "EAT1": "சாப்பிடு",
    "FATHER": "தந்தை",
    "FINISH": "முடி",
    "FRIENDLY": "நட்பான",
    "FUTURE": "எதிர்காலம்",
    "GO": "செல்",
    "GREEN": "பச்சை",
    "HELLO": "வணக்கம்",
    "HOSPITAL1": "மருத்துவமனை",
    "IMPORTANT": "முக்கியமான",
    "KING": "ராஜா",
    "LAUGH": "சிரி",
    "LEFT": "இடது",
    "MEDICINE": "மருந்து",
    "MOTHER": "தாய்",
    "MORNING": "காலை",
    "NIGHT1": "இரவு",
    "NOON1": "நண்பகல்",
    "PAIN": "வலி",
    "PERSON": "நபர்",
    "RAIN": "மழை",
    "RED": "சிவப்பு",
    "RIGHT2": "வலது",
    "RIVER1": "ஆறு",
    "SAME2": "ஒரே",
    "SCARED": "பயந்த",
    "SKILL": "திறமை",
    "SLOW": "மெதுவான",
    "SPORTS": "விளையாட்டுகள்",
    "STORY2": "கதை",
    "SURPRISE": "ஆச்சரியம்",
    "TEACH": "கற்பி",
    "THIRD1": "மூன்றாவது",
    "TIREDOF": "சலித்துவிட்ட",
    "TRAVEL": "பயணம்",
    "TWO": "இரண்டு",
    "UNDERSTAND": "புரிந்துகொள்",
    "UNIVERSITY": "பல்கலைக்கழகம்",
    "WANT1": "வேண்டும்",
    "WHAT1": "என்ன",
    "WHY": "ஏன்",
    "WOMAN1": "பெண்",
    "WORK": "வேலை",
    "YES": "ஆம்",
}

HINDI_CORRECTIONS = {
    "ADMIT": "स्वीकार करना",
    "ALL": "सभी",
    "AFTER": "बाद",
    "ACTION": "कार्य",
    "ABOUT1": "के बारे में",
    "ALARM": "अलार्म",
    "ANGRY": "गुस्सा",
    "APPLE": "सेब",
    "BATH": "स्नान",
    "BELIEVE1": "विश्वास करना",
    "BOY": "लड़का",
    "BREAKFAST1": "नाश्ता",
    "BROTHER": "भाई",
    "CANCER": "कैंसर",
    "DANCE": "नृत्य",
    "DARK1": "अंधेरा",
    "DINNER1": "रात का खाना",
    "DOG1": "कुत्ता",
    "EAT1": "खाना",
    "FINISH": "समाप्त करना",
    "FRIENDLY": "मिलनसार",
    "FUTURE": "भविष्य",
    "GO": "जाना",
    "GREEN": "हरा",
    "HELLO": "नमस्ते",
    "HOSPITAL1": "अस्पताल",
    "IMPORTANT": "महत्वपूर्ण",
    "KING": "राजा",
    "LAUGH": "हँसना",
    "LEFT": "बायाँ",
    "MEDICINE": "दवा",
    "MOTHER": "माँ",
    "MORNING": "सुबह",
    "NIGHT1": "रात",
    "NOON1": "दोपहर",
    "PAIN": "दर्द",
    "PERSON": "व्यक्ति",
    "RAIN": "बारिश",
    "RED": "लाल",
    "RIGHT2": "दायाँ",
    "RIVER1": "नदी",
    "SAME2": "समान",
    "SCARED": "डरा हुआ",
    "SKILL": "कौशल",
    "SLOW": "धीमा",
    "SPORTS": "खेल",
    "STORY2": "कहानी",
    "SURPRISE": "आश्चर्य",
    "THIRD1": "तीसरा",
    "TRAVEL": "यात्रा",
    "TWO": "दो",
    "UNDERSTAND": "समझना",
    "UNIVERSITY": "विश्वविद्यालय",
    "WANT1": "चाहना",
    "WHAT1": "क्या",
    "WHY": "क्यों",
    "WOMAN1": "महिला",
    "WORK": "काम",
}

# Apply corrections only where the class exists
for gloss in labels:
    if gloss in TAMIL_CORRECTIONS:
        translations["tamil"][gloss] = TAMIL_CORRECTIONS[gloss]

    if gloss in HINDI_CORRECTIONS:
        translations["hindi"][gloss] = HINDI_CORRECTIONS[gloss]

# Make sure every one of the 304 classes exists
for language in translations:
    for gloss in labels:
        if gloss not in translations[language]:
            translations[language][gloss] = gloss

# Save
with open(NEW_TRANSLATION_FILE, "wb") as f:
    pickle.dump(translations, f)

print("================================")
print("Translation file created!")
print("Classes:", len(labels))
print("File:", NEW_TRANSLATION_FILE)
print("================================")