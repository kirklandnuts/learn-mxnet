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


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# Main ----
if __name__ == "__main__":
    train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    # LINES 41 - 84 show two examples of implementing a MLP in blocks; they're
    #   commented out because their function is replicated in 6 lines below using
    #   mxnet sequential

    # # everything in mxnet is organized using blocks (layers, losses, whole networks)
    # # a block has one requirement -- it has a `forward` method that takes an NDArray
    # #   input and generates an NDArray output
    # class MLP(gluon.Block):
    #     def __init__(self, **kwargs):
    #         super(MLP, self).__init__(**kwargs)
    #         with self.name_scope():
    #             self.dense0 = gluon.nn.Dense(64)
    #             self.dense1 = gluon.nn.Dense(64)
    #             self.dense2 = gluon.nn.Dense(10)
    #
    #     def forward(self, x):
    #         x = nd.relu(self.dense0(x))
    #         x = nd.relu(self.dense1(x))
    #         x = self.dense2(x)
    #         return x
    #
    # # now, instantiate the perceptron
    # net = MLP()
    # # grab block's params with collect_params then initialize them
    # net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
    #
    # # debugging -- to see what's going on at each layer, insert print
    # #   statements in forward
    # class MLP(gluon.Block):
    #     def __init__(self, **kwargs):
    #         super(MLP, self).__init__(**kwargs)
    #         with self.name_scope():
    #             self.dense0 = gluon.nn.Dense(64, activation="relu")
    #             self.dense1 = gluon.nn.Dense(64, activation="relu")
    #             self.dense2 = gluon.nn.Dense(10)
    #
    #     def forward(self, x):
    #         x = self.dense0(x)
    #         print("Hidden Representation 1: %s" % x)
    #         x = self.dense1(x)
    #         print("Hidden Representation 2: %s" % x)
    #         x = self.dense2(x)
    #         print("Network output: %s" % x)
    #         return x
    #
    # net = MLP()
    # net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
    # net(data.as_in_context(model_ctx))


    # Faster modeling with gluon.nn.Sequential
    # the following code does the same as the two blocks above
    num_hidden = 64
    # instatiate a Sequential
    net = gluon.nn.Sequential()
    # adding layers; Sequential assumes that the layers arrive bottom to top
    #   (with input at the very bottom)
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

    epochs = 10
    smoothing_constant = .01

    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()


        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
