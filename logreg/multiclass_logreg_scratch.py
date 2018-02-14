"""
Multiclass logistic regression from scratch
Created following MXNet's The Straight Dope (http://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html)
"""


# Imports ----
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt



# Global constants ----
mx.random.seed(1)
num_inputs = 784
num_outputs = 10
num_examples = 60000
batch_size = 64
epochs = 5
learning_rate = .005


# Functions ----
def transform(data, label):
    # function for preprocessing data; casts data to floats and normalize to [0, 1]
    return data.astype(np.float32)/255, label.astype(np.float32)

def softmax(y_linear):
    # here, elementwise subtraction of the max value stabilizes the score
    #   before it is exponentiated; this keeps higher scores from corresponding
    #   to disproportionately high probabilities
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

def net(X):
    # model
    y_linear = nd.dot(X, W) + b
    # activation function
    yhat = softmax(y_linear)
    return yhat

def cross_entropy(yhat, y):
    # loss function
    return - nd.sum(y * nd.log(yhat+1e-6))

def SGD(params, lr):
    # stochastic gradient descent
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    # returns fraction of correct predictions / total number of questions
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        # argmax returns indices of maximum values along an axis
        # in this case, our model outputs probabilities for each of the 10
        # possible outcomes,
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

def model_predict(net,data):
    # returns integer prediction of what number
    output = net(data)
    return nd.argmax(output, axis=1)



# Main ----
if __name__ == '__main__':
    # setting contexts
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()

    # Downloading MNIST dataset
    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

    # Load data iterator
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    # Allocating model parameters
    # store weights W in 784 x 10 matrix; need 10 vectors of 784 weights to map
    #   each of the 784 features to each of the 10 output classes
    W = nd.random_normal(shape=(num_inputs, num_outputs),ctx=model_ctx)
    # store bias term in 10-d array
    b = nd.random_normal(shape=num_outputs,ctx=model_ctx)
    params = [W, b]

    # tell mxnet to expect gradients corresponding to these parameters
    for param in params:
        param.attach_grad()

    # pre-train accuracy
    print('pre-train accuracy: {}'.format(evaluate_accuracy(test_data, net)))

    # training loop
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1,784))
            label = label.as_in_context(model_ctx)
            # onehot: label == 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data)
                loss = cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)
            cumulative_loss += nd.sum(loss).asscalar()
        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e + 1, cumulative_loss/num_examples, train_accuracy, test_accuracy))

    # sample and predict upon 10 random data points from the test set
    sample_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
    for i, (data, label) in enumerate(sample_data):
        data = data.as_in_context(model_ctx)
        print(data.shape)
        im = nd.transpose(data,(1,0,2,3))
        im = nd.reshape(im,(28,10*28,1))
        imtiles = nd.tile(im, (1,1,3))

        plt.imshow(imtiles.asnumpy())
        plt.show()
        pred=model_predict(net,data.reshape((-1,784)))
        print('model predictions are:', pred)
        break
