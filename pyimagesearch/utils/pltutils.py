import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_loss_acc(H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    X = np.arange(0, len(H.history['loss']))
    plt.plot(X, H.history["loss"], label="train_loss")
    plt.plot(X, H.history["val_loss"], label="val_loss")
    plt.plot(X, H.history["acc"], label="train_acc")
    plt.plot(X, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def save_loss_acc(H, sava_path):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    X = np.arange(0, len(H.history['loss']))
    plt.plot(X, H.history["loss"], label="train_loss")
    plt.plot(X, H.history["val_loss"], label="val_loss")
    plt.plot(X, H.history["acc"], label="train_acc")
    plt.plot(X, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(sava_path)
    plt.close()


def plot_prob(preds, le, height=200, width=200):
    """
    Plot prediction probabilities.
    :param preds: predictions
    :param le: label encoder
    :return:
    """
    canvas = np.zeros((height, width, 3), dtype='uint8')

    for i, (prob, label) in enumerate(zip(preds, le.classes_)):
        text = '{}: {:.2f}%'.format(label, prob * 100)
        bar_height = height // len(le.classes_)
        cv2.rectangle(canvas, (0, i * bar_height), (int(prob * width), i * bar_height + bar_height), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * bar_height + bar_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
    return canvas
