import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import glob
import os

# Load all model history files
history_files = glob.glob('models_and_encoders/2training_history_*.pkl')  # Match all files with the pattern
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

# Sort the histories list by the number in ascending order
histories.sort(key=lambda x: x[1])
histories = np.array(histories)


# Plot Metrics
def plot_training_curves(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axs[0].plot(history['accuracy'], label='Train')
    axs[0].plot(history['val_accuracy'], label='Val')
    # axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(history['loss'], label='Train')
    axs[1].plot(history['val_loss'], label='Val')
    # axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def compare_augmentation(histories):
    """
    Compare the training curves of different models based on their training history.
    Arguments:
    histories -- list of tuples (history, number) where history is the training history and number is the model number
    Returns:
    None
    """
    xs = []
    ys = []
    for history, number in histories:
        loss = history['loss']
        val_loss = history['val_loss']

        xs.append(number)
        patience = 5
        count = 0
        for epoch in range(len(history['loss'])):
            if val_loss[epoch] > loss[epoch]:
                count += 1
            else:
                count = 0
            if count > patience:                
                ys.append(epoch-patience+1)
                break
            if epoch == len(history['loss']) - 1:
                ys.append(0)



        
    ax = plt.axes()
    # ax.set_title('Training Accuracy vs Augmentation Factor')
    ax.scatter(xs, ys, label='Overfited epoch')
    # plt.legend()
    plt.xlabel('Augmentation factor')
    plt.ylabel('Epoch causing overfitting')
    plt.show()


print("Keys in a history object:", list(histories[0][0].keys()))

plot_training_curves(histories[0][0])
# compare_augmentation(histories)
