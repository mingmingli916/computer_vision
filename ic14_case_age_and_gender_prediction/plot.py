import re
import matplotlib.pyplot as plt
import numpy as np


def mx_age_plot(network, dataset, log):
    train_acc, train_loss = [], []

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
        acc = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

        # update the batch training lists
        train_acc.append(float(acc))
        train_loss.append(float(loss))

    # extract the validation accuracies for each
    # epoch, followed by the loss
    val_acc = re.findall(r'Validation-accuracy=(.*)', rows)
    val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

    # convert the validation accuracy and loss lists to floats
    val_acc = [float(x) for x in val_acc]
    val_loss = [float(x) for x in val_loss]

    # one off accuracy
    train_one_off = re.findall(r'Train-one-off=(.*)', rows)
    test_one_off = re.findall(r'Test-one-off=(.*)', rows)

    # convert
    train_one_off = [float(x) for x in train_one_off]
    test_one_off = [float(x) for x in test_one_off]

    # plot the accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(start_epoch, start_epoch + len(train_acc)), train_acc, label="train_accuracy")
    plt.plot(np.arange(start_epoch, start_epoch + len(val_acc)), val_acc, label="val_accuracy")
    plt.title("{}: accuracy on {}".format(network, dataset))
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

    # plot the losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(start_epoch, start_epoch + len(train_one_off)), train_one_off, label="train one off")
    plt.plot(np.arange(start_epoch, start_epoch + len(test_one_off)), test_one_off, label="test one off")
    plt.title("{}: one-onf on {}".format(network, dataset))
    plt.xlabel("Epoch #")
    plt.ylabel("One Off")
    plt.legend(loc="upper right")
    plt.show()


def mx_gender_plot(network, dataset, log):
    train_acc, train_loss = [], []

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
        acc = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

        # update the batch training lists
        train_acc.append(float(acc))
        train_loss.append(float(loss))

    # extract the validation accuracies for each
    # epoch, followed by the loss
    val_acc = re.findall(r'Validation-accuracy=(.*)', rows)
    val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

    # convert the validation accuracy and loss lists to floats
    val_acc = [float(x) for x in val_acc]
    val_loss = [float(x) for x in val_loss]

    # plot the accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(start_epoch, start_epoch + len(train_acc)), train_acc, label="train_accuracy")
    plt.plot(np.arange(start_epoch, start_epoch + len(val_acc)), val_acc, label="val_accuracy")
    plt.title("{}: accuracy on {}".format(network, dataset))
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


if __name__ == '__main__':
    mx_gender_plot(
        'AgeNet',
        'adience',
        '/home/hack/PycharmProjects/computer_vision/ic14_case_age_and_gender_prediction/output/gender/training_0.log'
    )
