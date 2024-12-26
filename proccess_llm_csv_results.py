import pandas as pd
import os
import re

def find_number_in_string(s):
    # Match a number between 1 and 12 using a regex
    match = re.search(r'\b(1[0-2]|[1-9])\b', s)
    return int(match.group()) if match else None

# Define results file to process
prompt = ['list-reason', 'reason', 'txt', 'num']
model = ['llama-3.2-1b@q8', 'llama-3.2-3b@q4', 'llama-3.2-3b@q8', 'llama-3.2-8b@q4']

# Define the list of possible detected emotions
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise', 
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']

# Load the sentences & ground truth CSV file into a DataFrame
csv_df = pd.read_csv("tweet_emotions.csv")  # replace with the path to your CSV file

for p in prompt:
    for m in model:
        results_file = f"results_{p}_{m}.csv"
        if not os.path.exists(results_file):
            continue
        print(f"\nProcessing {results_file}")

        # Dictionary to store sentences and their detected emotions
        detected_emotions = {}
        with open(results_file, 'r') as file:
            not_detected = 0
            for line in file:
                try:
                    sentence, emotion_text = line.strip().split('","')
                except:
                    continue
                # Remove any leading/trailing quotes from the split strings
                sentence = sentence.strip('"')
                emotion_text = emotion_text.strip('"').lower()
                # Find the matched emotion from the list
                matched_emotion = None
                if p != "num":
                    matched_emotion = next((emotion for emotion in emotions if emotion in emotion_text), None)
                else:
                    index = find_number_in_string(emotion_text)
                    if index and 0 <= index < len(emotions):
                        matched_emotion = emotions[index]
                if not matched_emotion:
                    not_detected += 1
                    matched_emotion = "neutral"
                detected_emotions[sentence] = matched_emotion
        # Add detected emotion column to CSV DataFrame by matching sentences
        csv_df['Detected Emotion'] = csv_df['content'].map(detected_emotions)
        # Fill any missing detected emotions with "neutral"
        csv_df.fillna({'Detected Emotion': "neutral"}, inplace=True)
        # Save the updated DataFrame to a new CSV file
        csv_df.to_csv(f"processed_{p}_{m}.csv", index=False)
        print (f"Failed to detect emotion in {not_detected} sentences. Setting to neutral")
