import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, name, plot_val=True, patience=20):
    # Plot the re-trained MobileNetV2 training history
    """
    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'], label='Re-trained', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)
    """
    
    plt.figure(figsize=(15,5))
    plt.plot(history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
    if (plot_val):
        plt.plot(history['val_accuracy'], label='Val Accuracy', alpha=.8, color='#ff7f0e')
        plt.plot(history['val_precision'], label='Val Precision', alpha=.8, color='red')
        plt.plot(history['val_recall'], label='Val Recall', alpha=.8, color='blue')
    plt.legend(loc='upper left')
    plt.title(name + ' Accuracy')
    plt.grid(alpha=.3)

    # Add bottom text

    plt.text(0.5, -0.1, 'Train accuracy: ' + str(round(history['accuracy'][-patience-1], 4)) + ', Val accuracy: ' + str(round(np.max(history['val_accuracy']), 4)), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.show()