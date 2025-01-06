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

# Synonym-to-emotion mapping
synonym_to_emotion = {
    # Happiness-related synonyms
    "joy": "happiness", "cheerfulness": "happiness", "glee": "happiness", "bliss": "happiness", 
    "ecstasy": "happiness", "satisfaction": "happiness", "jubilation": "happiness",
    # Sadness-related synonyms
    "sorrow": "sadness", "grief": "sadness", "melancholy": "sadness", "mourning": "sadness", 
    "despair": "sadness", "heartache": "sadness", "loneliness": "sadness",
    # Fun-related synonyms
    "amusement": "fun", "enjoyment": "fun", "laughter": "fun", "playfulness": "fun", 
    "hilarity": "fun",
    # Anger-related synonyms
    "rage": "anger", "fury": "anger", "wrath": "anger", "irritation": "anger", 
    "resentment": "anger", "annoyance": "anger",
    # Love-related synonyms
    "affection": "love", "adoration": "love", "fondness": "love", "passion": "love", 
    "devotion": "love", "infatuation": "love",
    # Worry-related synonyms
    "fear": "worry", "anxiety": "worry", "concern": "worry", "unease": "worry", 
    "dread": "worry", "trepidation": "worry",
    # Surprise-related synonyms
    "shock": "surprise", "astonishment": "surprise", "amazement": "surprise", "awe": "surprise", 
    "startle": "surprise",
    # Relief-related synonyms
    "relaxed": "relief", "comfort": "relief", "ease": "relief", "reassurance": "relief", 
    "safety": "relief",
    # Hate-related synonyms
    "hatred": "hate", "loathing": "hate", "disgust": "hate", "abhorrence": "hate", 
    "revulsion": "hate",
    # Neutral-related synonyms
    "apathy": "empty", "indifference": "empty", "detachment": "empty", "numbness": "empty", 
    "unconcern": "empty",
}

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
                # Find the matched emotion from the list or synonyms
                matched_emotion = None
                if p != "num":
                    # First try to match exact emotions
                    matched_emotion = next((emotion for emotion in emotions if emotion in emotion_text), None)
                    if not matched_emotion:
                        # If no match, check synonyms
                        matched_emotion = next(
                            (synonym_to_emotion[synonym] for synonym in synonym_to_emotion if synonym in emotion_text), 
                            None
                        )
                else:
                    # For "num" prompts, map using the extracted number
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
        csv_df.to_csv(f"processed_with_synonym_matching_{p}_{m}.csv", index=False)
        print (f"Failed to detect emotion in {not_detected} sentences. Setting to neutral.")
