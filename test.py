import pandas as pd


df = pd.read_csv("paragraph_emotion_dataset.csv")

# Group by the 'sentiment' column and count the occurrences
emotion_counts = df['sentiment'].value_counts()

# Print the count for each emotion
print(emotion_counts)
