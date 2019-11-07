from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    # take the dot product between our features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points,
# where each data point is a 2D feature vector
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature matrix -- this little trick
# allows us to treat the bias as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of the data
# for training and the remaining 50% for testing
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []  # keep track of our losses

# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):
    preds = sigmoid_activation(train_x.dot(W))
    error = preds - train_y
    loss = np.sum(error ** 2)
    losses.append(loss)

    gradient = train_x.T.dot(error)  # because of loss definition
    # gradient descent
    W += -args['alpha'] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print('[INFO] epoch={}, loss={:.7f}'.format(int(epoch + 1), loss))

preds = predict(test_x, W)
print(classification_report(test_y, preds))

# plot the testing classification data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(test_x[:, 0], test_x[:, 1], marker="o", c=test_y.reshape(-1))

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
