import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer
        # but stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the number of nodes
            # in each respective layer together, adding an extra node for the bias
            # The matrix is M x N since we wish to connect every node in current layer to
            # every node in the next layer.
            w = np.random.rand(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case where the input connections
        # need a bias term but the output does not
        w = np.random.rand(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return 'NeuralNetwork: {}'.format('-'.join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # activation function
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        # insert a column of 1's as the last entry in the feature matrix --
        # this little trick allows us to treat the bias as a trainable parameter
        # within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs):
            # loop over each individual data point and train our network on it
            for x, target in zip(X, y):
                self.fit_partial(x, target)

            if epoch % display_update == 0:
                loss = self.calculate_loss(X, y)
                print('[INFO] epoch={}, loss={:.7f}'.format(epoch, loss))

    def fit_partial(self, x, y):
        # construct our list of output activation for each layer
        # as our dat point flows through the network; the first
        # activation is a special case -- it's just the input
        # feature vector itself
        # This is responsible for storing the output activations for
        # each layers as our data point x forward propagates through the
        # network.
        # We initialize this list with x, which is simply the input data point.
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        for layer in range(len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropapation is to compute the
        # difference between our prediction (the final output
        # activation in the activation list) and the true target value
        err = A[-1] - y

        # from here, we need to apply the chain rule and build our
        # list of delta 'D'; the first entry in the delta is simply
        # the error of the output layer times the derivative of
        # our activation function for the output value
        D = [err * self.sigmoid_deriv(A[-1])]

        # simply loop over the layers in reverse order (ignoring the
        # last two since we already have taken them into account)
        for layer in range(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the previous layer dotted with weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)  # derivative
            delta = delta * self.sigmoid_deriv(A[layer])  # notice the order
            D.append(delta)

        # sine we looped over our layers in reverse order we need to
        # reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in range(len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning"
            # takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        # initialize the output prediction as the input feature -- this
        # Value will be (forward) propagated through the network to obtain
        # the final prediction
        p = np.atleast_2d(X)

        if add_bias:
            # insert a column of 1's as the last entry in the feature matrix
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in range(len(self.W)):
            # computing the output prediction is as simple as taking
            # the dot product between the current activation value 'p'
            # and the weight matrix associated with the current activation
            # function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        # make prediction for the input data points then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = .5 * np.sum((predictions - targets) ** 2)
        return loss
