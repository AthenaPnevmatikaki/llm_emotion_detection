import pandas as pd
import re

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
    "excitement": "fun",
    
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
csv_df = pd.read_csv("paragraph_emotion_dataset.csv")  # replace with the path to your CSV file

# Dictionary to store sentences and their detected emotions
detected_emotions = {}

# Regular expression pattern to match sentences and detected emotions
sentence_pattern = re.compile(r'{"sentence":\s*"(.*?)"')
emotion_pattern = re.compile(r'\b(' + '|'.join(emotions) + r')\b', re.IGNORECASE)

# Open the text file and process line by line
results_file = "paragraph_results_llama-3.2-1b-instruct.txt"
with open(results_file, 'r') as file:  # replace with the path to your TXT file
    current_sentence = None  # Track the current sentence being processed
    buffer = []  # Store lines related to the current sentence
    not_detected = 0

    for line in file:
        # Check if the line contains a new sentence
        sentence_match = sentence_pattern.search(line)
        if sentence_match:
            # Process the previous sentence buffer to detect emotion
            if current_sentence:
                # Join buffer lines and search for the first emotion occurrence
                text_block = " ".join(buffer)
                emotion_match = emotion_pattern.search(text_block)
                if emotion_match:
                    detected_emotion = emotion_match.group(1).lower()
                else:
                    # Check synonyms if no direct match is found
                    detected_emotion = next(
                        (synonym_to_emotion[synonym] for synonym in synonym_to_emotion if synonym in text_block), 
                        None
                    )
                    if not detected_emotion:
                        detected_emotion = "neutral"
                        not_detected += 1
                
                detected_emotions[current_sentence] = detected_emotion
                

            # Update current sentence and reset the buffer
            current_sentence = sentence_match.group(1)
            buffer = []
        else:
            # If not a new sentence, add the line to the buffer
            buffer.append(line)
        
    # Handle the last sentence in the buffer
    if current_sentence:
        text_block = " ".join(buffer)
        emotion_match = emotion_pattern.search(text_block)
        if emotion_match:
            detected_emotion = emotion_match.group(1).lower()
            detected_emotions[current_sentence] = detected_emotion
        else:
            detected_emotions[current_sentence] = "neutral"
            detected_emotion = "neutral"
            not_detected += 1

# Add detected emotion column to CSV DataFrame by matching sentences
csv_df['Detected Emotion'] = csv_df['content'].map(detected_emotions)

# Fill any missing detected emotions with "neutral"
csv_df.fillna({'Detected Emotion': "neutral"}, inplace=True)

# Save the updated DataFrame to a new CSV file
csv_df.to_csv(f"processed_{results_file}.csv", index=False)

print (f"Failed to detect emotion in {not_detected} sentences. Setting to neutral")
print("CSV file has been updated with detected emotions.")
