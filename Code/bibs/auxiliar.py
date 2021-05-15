import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_results(training, y_test_pred, y_test, classes):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # ***************************************************************************************************************************
    # Plot Training
    ax1.plot(training.history['sparse_categorical_accuracy'])
    # ax1.plot(training.history['val_sparse_categorical_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    # ***************************************************************************************************************************
    # Confusion matrix
    y_pred = []

    for i in y_test_pred:
        y_pred.append(i.argmax())

    y_pred = np.asarray(y_pred)

    matrix = confusion_matrix(y_test, y_pred)

    im = ax2.imshow(matrix)

    # We want to show all ticks...
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax2.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    ax2.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()