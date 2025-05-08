import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import glob
import os

# Load all model history files
history_files = glob.glob('models_and_encoders/training_history_*.pkl')  # Match all files with the pattern
histories = []  # List to store histories and their corresponding numbers

for file_path in history_files:
    # Extract the number at the end of the file name
    file_name = os.path.basename(file_path)
    number = int(file_name.split('_')[-1].split('.')[0])  # Extract the number before the file extension

    # Load the history
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    
    # Append the history and its number as a tuple
    histories.append((history, number))
histories = np.array(histories)


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
def compare_augmentation(histories):
    """
    Compare the training curves of different models based on their training history.
    Arguments:
    histories -- list of tuples (history, number) where history is the training history and number is the model number
    Returns:
    None
    """
    ax = plt.axes()
    for history, number in histories:
        ax.plot(history['val_loss'], label=f'Augmentation factor: {number}')
    ax.set_title('Training Accuracy vs Augmentation Factor')
    plt.legend()
    plt.xlabel('Epoch')

    plt.show()

# plot_training_curves(history)
compare_augmentation(histories)