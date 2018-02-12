# Imports ----
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import matplotlib.pyplot as plt


# Global constants ----
batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000
epochs = 10


# Functions ----
def transform(data, label):
    # function for preprocessing data; casts data to floats and normalize to [0, 1]
    return data.astype(np.float32)/255, label.astype(np.float32)

def evaluate_accuracy(data_iterator, net):
    # returns accuracy as calculated by mx metric package
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        prediction = nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]

def model_predict(net,data):
    output = net(data.as_in_context(model_ctx))
    return nd.argmax(output, axis=1)


# Main ----
if __name__ == "__main__":
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()

    # Downloading MNIST dataset
    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

    # Load data iterator
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    # define model
    net = gluon.nn.Dense(num_outputs)

    # initialize parameters
    net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

    # softmax cross entropy loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # instantiate optimizer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    # pre-train accuracy
    print('pre-train accuracy: {}'.format(evaluate_accuracy(test_data, net)))

    # training loop
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1,784))
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

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
