import pandas as pd
import re

# Define the list of possible detected emotions
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise', 
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']

# Load the CSV file into a DataFrame
csv_df = pd.read_csv("tweet_emotions.csv")  # replace with the path to your CSV file

# Dictionary to store sentences and their detected emotions
detected_emotions = {}

# Regular expression pattern to match sentences and detected emotions
sentence_pattern = re.compile(r'{"sentence":\s*"(.*?)"')
emotion_pattern = re.compile(r'\b(' + '|'.join(emotions) + r')\b', re.IGNORECASE)

# Open the text file and process line by line
results_file = "results_reason_llama-3.2-1b@q8.txt"
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
                    detected_emotions[current_sentence] = detected_emotion
                else:
                    detected_emotions[current_sentence] = "neutral"
                    detected_emotion = "neutral"
                    not_detected += 1

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
