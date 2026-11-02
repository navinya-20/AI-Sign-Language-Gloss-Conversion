
from translation import translate_all_languages

word = "ADMIT"

result = translate_all_languages(word)

print("\n==============================")
print("      5 LANGUAGE OUTPUT")
print("==============================")

for language, translation in result.items():
    print(f"{language:10} : {translation}")

print("==============================")

