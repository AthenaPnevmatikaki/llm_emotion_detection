import pandas as pd
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import os

prompts = ['list-reason', 'reason', 'txt', 'num']
models = ['llama-3.2-1b@q8', 'llama-3.2-3b@q4', 'llama-3.2-3b@q8', 'llama-3.2-8b@q4',
          'emollama-3.1-8b@q4', 'emollama-chat-7b@q4', 'emollama-chat-13b@q3',
          'bert-7b@q6', 'bloomz-7b@q6', 'vicuna-13b@q3', 'TFIDF', 'word2vec', 'word2vec-nn']
# Use None to score every row in the prompt-model combination, or
# the name of a CSV (e.g. processed_txt_word2vec.csv) to score only against the results found there
guide = 'processed_txt_word2vec-nn.csv'
grouping = 1 # 0: No grouping, 1: 1st-level grouping into 7 classes, 2: 2nd-level grouping into 3 classes
show_cm = False
scores_name = 'Scores'
if guide:
    scores_name= f'{scores_name} 8k'
if grouping > 0:
    scores_name= f'{scores_name} level {grouping} grouping'
scores_name= f'{scores_name}.csv'
# List of emotions to keep the order consistent
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']
emotion_grouping = None
if grouping == 2:
    emotion_grouping = {
        "sadness": "Negative",
        "empty": "Negative",
        "boredom": "Negative",
        "worry": "Negative",
        "anger": "Negative",
        "hate": "Negative",
        "neutral": "Neutral",
        "relief": "Neutral",
        "surprise": "Neutral",
        "happiness": "Positive",
        "love": "Positive",
        "enthusiasm": "Positive",
        "fun": "Positive",
    }
if grouping == 1:
    emotion_grouping = {
        'sadness': 'Worry & Sadness',
        'worry': 'Worry & Sadness',
        'boredom': 'Worry & Sadness',
        'fun': 'Joy & Excitement',
        'enthusiasm': 'Joy & Excitement',
        'happiness': 'Joy & Excitement',
        'love': 'Love',
        'relief': 'Neutral',
        'anger': 'Hostility',
        'hate': 'Hostility',
        'surprise': 'Surprise',
        'neutral': 'Neutral',
        'empty': 'Worry & Sadness'
    }
if emotion_grouping is not None:
    emotions = list(set(emotion_grouping.values()))

if guide:
    guide = pd.read_csv(f"processed_results/{guide}")
scores = []
for prompt in prompts:
    for model in models:
        if not os.path.exists(f"processed_results/processed_{prompt}_{model}.csv"):
            continue
        print(f"\nScoring {prompt}_{model}")
        data = pd.read_csv(f"processed_results/processed_{prompt}_{model}.csv")
        if guide is not None:
            data = data[data['tweet_id'].isin(guide['tweet_id'])]
        if emotion_grouping is not None:
            data["sentiment"] = data["sentiment"].map(emotion_grouping)
            data["Detected Emotion"] = data["Detected Emotion"].map(emotion_grouping)
        # Define the ground truth and detected emotion columns
        y_true = data["sentiment"]
        y_pred = data["Detected Emotion"]
        if show_cm:
            # Compute the confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=emotions)
            # Display the confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axes
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)
            disp.plot(cmap="Blues", xticks_rotation="vertical", ax=ax)  # Plot on the specified axes
            # Highlight the diagonal
            for i in range(len(emotions)):  # Loop through the diagonal
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, edgecolor="red", fill=False, lw=2))
            plt.title(f"{model} using prompt {prompt}", fontsize=16)
            plt.tight_layout()
            plt.show()
        # Metric scores
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        # Count the number of unique clusters (unique emotions predicted)
        unique_clusters = data["Detected Emotion"].nunique()
        print(f"Number of clusters (unique predicted emotions): {unique_clusters}")
        scores.append({
                "Model": model,
                "Prompt": prompt,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1})
scores = pd.DataFrame(scores)
print(scores)
scores.to_csv(scores_name, index=False)
