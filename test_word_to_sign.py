from word_to_sign import find_gloss, get_videos_for_gloss


# English test
gloss = find_gloss("ADMIT", "english")

print()
print("English:")
print("Gloss:", gloss)

videos = get_videos_for_gloss(gloss)

print("Videos found:", len(videos))

for video in videos[:3]:
    print(video)


# Tamil test
gloss = find_gloss("வழிகாட்டி", "tamil")

print()
print("Tamil:")
print("Gloss:", gloss)

videos = get_videos_for_gloss(gloss)

print("Videos found:", len(videos))

for video in videos[:3]:
    print(video)


# Telugu test
gloss = find_gloss("అడ్మిట్", "telugu")

print()
print("Telugu:")
print("Gloss:", gloss)

videos = get_videos_for_gloss(gloss)

print("Videos found:", len(videos))

for video in videos[:3]:
    print(video)