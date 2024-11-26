import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("processed_results_reason_llama-3.2-1b-instruct.txt.csv")

# Define the ground truth and detected emotion columns
y_true = data["sentiment"]
y_pred = data["Detected Emotion"]

# List of emotions to keep the order consistent
emotions = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise', 
            'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']

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