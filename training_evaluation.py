import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

# Load the model history
history_path = 'models_and_encoders/training_history.pkl'  # Path to the model history file
with open(history_path, 'rb') as f:
    history = pickle.load(f)
print(history.keys())  # Check the keys in the history dictionary

# Plot Metrics
def plot_training_curves(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axs[0].plot(history['accuracy'], label='Train')
    axs[0].plot(history['val_accuracy'], label='Val')
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(history['loss'], label='Train')
    axs[1].plot(history['val_loss'], label='Val')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix using sklearn's confusion_matrix and ConfusionMatrixDisplay.
    Arguments:
    y_true -- true labels
    y_pred -- predicted labels
    labels -- list of class labels
    Returns:
    None
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

plot_training_curves(history)
