import pandas as pd
import requests
import json
import time
import re

model = 'emollama-3.1-8b@q4'
prompt = 'txt'
filetype = 'csv'
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']
if prompt == 'list-reason':
    max_tokens = 1000
    content = """Return the emotion in the given sentences.
        Select between empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.
        Reply only with a list of the emotions you are most confident that are in the text, their strength and the reasons you selected them.
        Structure the returned list as JSON using: [{"emotion": "detected emotion", "strength": "medium/high", "rationale": "your explanation"]}, ...]"""
elif prompt == 'reason':
    max_tokens = 1000
    content = """Return the most evident emotion in the given sentences.
        Reply only with one emotion from this list: sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.
        Include your reason for selecting this emotion.
        Structure the returned list as JSON using: [{"emotion": "detected emotion", "rationale": "your explanation"]}, ...]"""
elif prompt == 'txt':
    max_tokens = 10
    content = """Detect the most prominent emotion in the sentence I will give you, by selecting the emotion from the following list:
        sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, anger
        Your reply must be just one word selected from the above list. Do not use anything else!"""
else:
    max_tokens = 1
    content = """Detect the most prominent emotion expressed in the text I will give you.
        Use only a number from the list 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12 in your response to represent the detected emotion, where the returned numbers
        correspond to detected emotions using the following mapping:
            1. sadness
            2. enthusiasm
            3. neutral
            4. worry
            5. surprise
            6. love
            7. fun
            8. hate
            9.happiness
            10. boredom
            11. relief
            12. anger
        Do not use any words in your response, just the number corresponding to the most prominent emotion"""
url = "http://localhost:1234/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
if filetype == 'csv':
    df = pd.read_csv(f"results_{prompt}_{model}.csv", header=None, names=["phrase", "emotion"])
else:
    exit(-1)

def contains_valid_emotion_txt(emotion):
    return any(e in emotion for e in emotions)

def contains_valid_emotion_num(emotion):
    match = re.search(r'\b(1[0-2]|[1-9])\b', emotion)
    index = int(match.group()) if match else None
    if index and 0 <= index < len(emotions):
        return True
    return False

def calculate_new_emotion(phrase, current_index, total_count, start_time):
    data = {"model": model, "messages": []}
    data["messages"].append({"role": "system", "content": content})
    data["messages"].append({"role": "user", "content": phrase})
    data["max_tokens"] = max_tokens
    data["temperature"] = 0.2
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    response_text = response['choices'][0]['message']['content']
    start_idx = response_text.find('[')
    end_idx = response_text.find(']', start_idx) + 1
    if start_idx != -1 and end_idx != -1:
        response_text = response_text[start_idx:end_idx]
    elapsed_time = time.time() - start_time
    average_time = elapsed_time / (current_index + 1)
    remaining_time = average_time * (total_count - current_index - 1) / 3600
    if current_index % 10 == 0:
        print(f"{current_index + 1}/{total_count}: Time per phrase {average_time:.3f}, {remaining_time:.3f} hours remaining")
    return response_text

if prompt != 'num':
    rows_to_process = df[~df["emotion"].apply(contains_valid_emotion_txt)]
else:
    rows_to_process = df[~df["emotion"].apply(contains_valid_emotion_num)]
total_to_process = len(rows_to_process)
start_time = time.time()
df.loc[rows_to_process.index, "emotion"] = rows_to_process.apply(
    lambda row: calculate_new_emotion(
        row["phrase"],
        rows_to_process.index.get_loc(row.name),
        total_to_process,
        start_time
    ),
    axis=1
)
if filetype == 'csv':
    df.to_csv(f"results_{prompt}_{model}.csv", index=False, header=False)
