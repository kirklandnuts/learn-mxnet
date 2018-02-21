# Librarys ----
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon


# Global constants ----
# memory contexts
data_ctx = mx.cpu()
model_ctx = mx.cpu()
# data params
num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
# training params
epochs = 10
learning_rate = .001
smoothing_constant = .01


# Functions ----
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


def relu(X):
    return nd.maximum(X, nd.zeros_like(X))


def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition


def softmax_cross_entropy(yhat_linear, y):
    # http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-scratch.html#The-softmax-cross-entropy-loss-function
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)


# Main ----
if __name__ == "__main__":
    train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    # Set some constants so it's easy to modify the network later
    num_hidden = 256
    weight_scale = .01

    # Allocate parameters for the first hidden layer
    W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
    b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

    # Allocate parameters for the second hidden layer
    W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
    b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

    # Allocate parameters for the output layer

    W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
    b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

    params = [W1, b1, W2, b2, W3, b3]

    # Allocate space for each parameter's gradients
    for param in params:
        param.attach_grad()

    # define model
    def net(X):
        # first hidden layer
        h1_linear = nd.dot(X, W1) + b1
        h1 = relu(h1_linear)

        # second hidden layer
        h2_linear = nd.dot(h1, W2) + b2
        h2 = relu(h2_linear)

        # output layer
        # no softmax function bc it will be applied in the softmax_cross_entropy loss
        yhat_linear = nd.dot(h2, W3) + b3
        return yhat_linear

    # training loop
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)
            cumulative_loss += nd.sum(loss).asscalar()


        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


    # sample 10 random data points from the test set
    samples = 10
    sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                            samples, shuffle=True)
    for i, (data, label) in enumerate(sample_data):
        data = data.as_in_context(model_ctx)
        im = nd.transpose(data,(1,0,2,3))
        im = nd.reshape(im,(28,10*28,1))
        imtiles = nd.tile(im, (1,1,3))

        plt.imshow(imtiles.asnumpy())
        plt.show()
        pred=model_predict(net,data.reshape((-1,784)))
        print('model predictions are:', pred)
        print('true labels :', label)
        break
