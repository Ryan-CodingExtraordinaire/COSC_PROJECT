import os
import json
import matplotlib.pyplot as plt
def check():
    # Load the data
    data_dir = "landmarks_data"  # Update with the correct directory if needed
    data = []

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                data.append(json_data)  # Append each JSON object to the list

    # Check if the 'label' key exists in the data
    if not all('label' in item for item in data):
        raise ValueError("Some entries in the dataset do not contain a 'label' key.")


    # Plot the distribution of labels
    labels = [item['label'] for item in data]
    ax = plt.axes()
    ax.bar(labels, [labels.count(label) for label in labels], color='blue', alpha=0.7)
    # Add labels and title
    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')
    # ax.set_title('Distribution of Labels')
    plt.tight_layout()

    # Show the plot
    plt.show()