import pandas as pd

# Define the list of possible detected emotions
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise', 
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']

# Synonym-to-emotion mapping
synonym_to_emotion = {
    # Happiness-related synonyms
    "joy": "happiness",
    "cheerfulness": "happiness",
    "glee": "happiness",
    "bliss": "happiness",
    "ecstasy": "happiness",
    "satisfaction": "happiness",
    "jubilation": "happiness",
    
    # Sadness-related synonyms
    "sorrow": "sadness",
    "grief": "sadness",
    "melancholy": "sadness",
    "mourning": "sadness",
    "despair": "sadness",
    "heartache": "sadness",
    "loneliness": "sadness",
    
    # Fun-related synonyms
    "amusement": "fun",
    "enjoyment": "fun",
    "laughter": "fun",
    "playfulness": "fun",
    "hilarity": "fun",
    
    # Anger-related synonyms
    "rage": "anger",
    "fury": "anger",
    "wrath": "anger",
    "irritation": "anger",
    "resentment": "anger",
    "annoyance": "anger",
    
    # Love-related synonyms
    "affection": "love",
    "adoration": "love",
    "fondness": "love",
    "passion": "love",
    "devotion": "love",
    "infatuation": "love",
    
    # Worry-related synonyms
    "fear": "worry",
    "anxiety": "worry",
    "concern": "worry",
    "unease": "worry",
    "dread": "worry",
    "trepidation": "worry",
    
    # Surprise-related synonyms
    "shock": "surprise",
    "astonishment": "surprise",
    "amazement": "surprise",
    "awe": "surprise",
    "startle": "surprise",
    
    # Relief-related synonyms
    "relaxed": "relief",
    "comfort": "relief",
    "ease": "relief",
    "reassurance": "relief",
    "safety": "relief",
    
    # Hate-related synonyms
    "hatred": "hate",
    "loathing": "hate",
    "disgust": "hate",
    "abhorrence": "hate",
    "revulsion": "hate",
    
    # Neutral-related synonyms
    "apathy": "empty",
    "indifference": "empty",
    "detachment": "empty",
    "numbness": "empty",
    "unconcern": "empty",
}

# Load the CSV file into a DataFrame
csv_df = pd.read_csv("tweet_emotions.csv")  # replace with the path to your CSV file

# Dictionary to store sentences and their detected emotions
detected_emotions = {}

# Open the text file and process line by line
results_file = "results_llama-3.2-1b-instruct.csv"
with open(results_file, 'r') as file:  # replace with the path to your TXT file
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
        matched_emotion = next((emotion for emotion in emotions if emotion in emotion_text), None)
        if not matched_emotion:
            # Check synonyms if no direct match is found
            matched_emotion = next(
                (synonym_to_emotion[synonym] for synonym in synonym_to_emotion if synonym in emotion_text), 
                None
            )
        if not matched_emotion:
            not_detected += 1
            matched_emotion = "neutral"
        detected_emotions[sentence] = matched_emotion

# Add detected emotion column to CSV DataFrame by matching sentences
csv_df['Detected Emotion'] = csv_df['content'].map(detected_emotions)

# Fill any missing detected emotions with "neutral"
csv_df.fillna({'Detected Emotion': "neutral"}, inplace=True)

# Save the updated DataFrame to a new CSV file
csv_df.to_csv(f"processed_{results_file}", index=False)

print (f"Failed to detect emotion in {not_detected} sentences. Setting to neutral")
print("CSV file has been updated with detected emotions.")
