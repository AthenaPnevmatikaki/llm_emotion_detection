import pandas as pd
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Processed results files to fuse. The first one dictates the sentences that are to be used
include_8k = True
if not include_8k:
    results_initial = {'processed_txt_bert-7b@q6.csv': 0.232,
                       'processed_txt_llama-3.2-8b@q4.csv': 0.224,
                       'processed_txt_emollama-chat-13b@q3.csv': 0.1846}
    weight_exponents = [1, 2, 3, 4, 5, 6, 7]
else:
    results_initial = {'processed_txt_word2vec-nn.csv': 0.165,
                       'processed_txt_bert-7b@q6.csv': 0.229,
                       'processed_txt_llama-3.2-8b@q4.csv': 0.213,
                       'processed_txt_emollama-chat-13b@q3.csv': 0.176}
    weight_exponents = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
grouping = 2 # 0: No grouping, 1: 1st-level grouping into 7 classes, 2: 2nd-level grouping into 3 classes
show_cm = False
scores_name = 'Scores'
plot_title = 'Fusing'
if include_8k:
    scores_name= f'{scores_name} 8k'
    plot_title = f'{plot_title} 8k'
if grouping > 0:
    scores_name= f'{scores_name} level {grouping} grouping'
    plot_title= f'{plot_title} level {grouping} grouping'
scores_name= f'{scores_name} fused.csv'
plot_title = f'{plot_title} results'
# List of emotions to keep the order consistent
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']
model_columns = list(results_initial.keys())
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

# Function for weighted voting
def weighted_voting(row, models, weights):
    vote_count = defaultdict(float)  # Dictionary to store weighted votes
    for model in models:
        emotion = row[model]
        vote_count[emotion] += weights[model]  # Add the model's weight to the emotion
    # Return the emotion with the highest weighted vote
    return max(vote_count, key=vote_count.get)

for i, result in enumerate(model_columns):
    if i == 0:
        data = pd.read_csv(f"processed_results/{result}")
        data[result] = data['Detected Emotion']
        del data['Detected Emotion']
        if emotion_grouping is not None:
            data["sentiment"] = data["sentiment"].map(emotion_grouping)
    else:
        tmp = pd.read_csv(f"processed_results/{result}")
        data = data.merge(
            tmp[["tweet_id", "Detected Emotion"]],  # Select only relevant columns from tmp
            on="tweet_id",  # Merge on tweet_id
            how="left",  # Keep all rows from data
            suffixes=("", "_tmp")  # Handle column name conflicts if necessary
        )
        data.rename(columns={"Detected Emotion": result}, inplace=True)
        del tmp
    if emotion_grouping is not None:
        data[result] = data[result].map(emotion_grouping)
del data['content']
del data['tweet_id']

scores = []
for weight_exponent in weight_exponents:
    results = {model: np.pow(weight,weight_exponent) for model, weight in results_initial.items()}
    total_weight = sum(results.values())
    results = {model: weight / total_weight for model, weight in results.items()}
    print(results)
    data["Fused Prediction"] = data.apply(weighted_voting, axis=1, models=model_columns, weights=results)
    # Define the ground truth and detected emotion columns
    y_true = data["sentiment"]
    y_pred = data["Fused Prediction"]
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
        plt.title("Fused decisions", fontsize=16)
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
    unique_clusters = data["Fused Prediction"].nunique()
    print(f"Number of clusters (unique predicted emotions): {unique_clusters}")
    scores.append({"Weight exponent": weight_exponent,
                   "Max weight": max(results.values()),
                   "Accuracy": acc,
                   "Precision": precision,
                   "Recall": recall,
                   "F1-Score": f1})
scores = pd.DataFrame(scores)
print(scores)
scores.to_csv(scores_name, index=False)
plt.figure(figsize=(8, 5))  # Optional: Set figure size
plt.plot(scores["Max weight"], scores["Recall"], marker='o', linestyle='-', color='b', label="Fusion")
plt.axhline(y=scores["Recall"][len(scores) - 1], color='r', linestyle='--', label="Bert")
plt.xlabel("Weight of most trusted model (Bert)")
plt.ylabel("Balanced accuracy")
plt.title(plot_title)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()