import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import os

prompt = ['list-reason', 'reason', 'txt', 'num']
model = ['llama-3.2-1b@q8', 'llama-3.2-3b@q4', 'llama-3.2-3b@q8', 'llama-3.2-8b@q4']

# List of emotions to keep the order consistent
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']

for p in prompt:
    for m in model:
        if not os.path.exists(f"processed_{p}_{m}.csv"):
            continue
        print(f"\nScoring {p}_{m}")
        data = pd.read_csv(f"processed_{p}_{m}.csv")
        # Define the ground truth and detected emotion columns
        y_true = data["sentiment"]
        y_pred = data["Detected Emotion"]
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=emotions)
        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)
        disp.plot(cmap="Blues", xticks_rotation="vertical")
        plt.title("Confusion Matrix of Detected vs. True Emotions")
        plt.show()
        # Compute the balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        # Compute the regular accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        # Count the number of unique clusters (unique emotions predicted)
        unique_clusters = data["Detected Emotion"].nunique()
        print(f"Number of clusters (unique predicted emotions): {unique_clusters}")
