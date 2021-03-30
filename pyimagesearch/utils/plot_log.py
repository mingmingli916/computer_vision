import matplotlib.pyplot as plt
import numpy as np
import re


def mx_plot(network, dataset, log):
    train_rank1, train_rank5, train_loss = [], [], []

    # load the contents of the log, then initialize the batch lists for
    # the training and validation data
    rows = open(log).read().strip()

    # grab the set of training epochs
    epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
    epochs = sorted([int(e) for e in epochs])

    start_epoch = min(epochs)

    for e in epochs:
        # find all rank-1 accuracies, rank-5 accuracies, and loss
        # values, then take the final entry in the list for each

        s = r'Epoch\[' + str(e) + '\].*accuracy=([0]*\.?[0-9]+)'
        rank1 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*top_k_accuracy_5=([0]*\.?[0-9]+)'
        rank5 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

        # update the batch training lists
        train_rank1.append(float(rank1))
        train_rank5.append(float(rank5))
        train_loss.append(float(loss))

    # extract the validation rank-1 and rank-5 accuracies for each
    # epoch, followed by the loss
    val_rank1 = re.findall(r'Validation-accuracy=(.*)', rows)
    val_rank5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
    val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

    # convert the validation rank-1, rank-5, and loss lists to floats
    val_rank1 = [float(x) for x in val_rank1]
    val_rank5 = [float(x) for x in val_rank5]
    val_loss = [float(x) for x in val_loss]

    # plot the accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(start_epoch, start_epoch + len(train_rank1)), train_rank1, label="train_rank1")
    plt.plot(np.arange(start_epoch, start_epoch + len(train_rank5)), train_rank5, '--', label="train_rank5")
    plt.plot(np.arange(start_epoch, start_epoch + len(val_rank1)), val_rank1, label="val_rank1")
    plt.plot(np.arange(start_epoch, start_epoch + len(val_rank5)), val_rank5, '--', label="val_rank5")
    plt.title("{}: rank-1 and rank-5 accuracy on {}".format(network, dataset))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # plot the losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(start_epoch, start_epoch + len(train_loss)), train_loss, label="train_loss")
    plt.plot(np.arange(start_epoch, start_epoch + len(val_loss)), val_loss, label="val_loss")
    plt.title("{}: cross-entropy loss on {}".format(network, dataset))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()
