import json
from collections import Counter

with open('raw_dataset/nslt_100.json', 'r') as f:
    nslt_data = json.load(f)

with open('raw_dataset/WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

# nslt_100.json maps video_id -> {action: [class_id, start_frame, end_frame], split: ...}
# WLASL_v0.3.json maps gloss (word) -> list of instances (each with video_id)

# Build video_id -> word lookup from WLASL_v0.3.json
video_id_to_word = {}
for entry in wlasl_data:
    word = entry['gloss']
    for instance in entry['instances']:
        video_id_to_word[instance['video_id']] = word

# Count how many nslt_100 videos we actually have on disk per word
import os
available_videos = set(f.replace('.mp4', '') for f in os.listdir('raw_dataset/videos'))

word_counts = Counter()
for video_id in nslt_data.keys():
    if video_id in available_videos and video_id in video_id_to_word:
        word = video_id_to_word[video_id]
        word_counts[word] += 1

print(f"Total distinct words in nslt_100 with available video files: {len(word_counts)}")
print("\nTop 50 words by available video count:")
for word, count in word_counts.most_common(50):
    print(f"  {word}: {count} videos")
