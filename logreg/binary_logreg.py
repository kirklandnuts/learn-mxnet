# Imports ----
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib
import matplotlib.pyplot as plt


# Global variables ----
data_ctx = mx.cpu()
model_ctx = mx.cpu()
batch_size = 64
epochs = 30
learning_rate = 0.02

# Functions ----
def logistic(z):
    return 1. / (1. + nd.exp(-z))


def process_data(raw_data):
    """processes raw data into X and Y ndarrays"""
    lines = raw_data.splitlines()
    num_examples = len(lines)
    num_features = 123
    X = nd.zeros((num_examples, num_features), ctx=data_ctx)
    Y = nd.zeros((num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(lines):
        tokens = line.split()
        label = (int(tokens[0]) + 1) / 2
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y

def log_loss(output, y):
    yhat = logistic(output)
    return - nd.nansum(y * nd.log(yhat) + (1-y) * nd.log(1-yhat))


# Main ----
# load data
with open('./data/a1a.train') as f:
    train_raw = f.read()
with open('./data/a1a.test') as f:
    test_raw = f.read()
Xtrain, Ytrain = process_data(train_raw)
Xtest, Ytest = process_data(test_raw)

# instantiate a data loader

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, Ytrain),
                                    batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, Ytest),
                                    batch_size=batch_size, shuffle=True)

# define model
net = gluon.nn.Dense(1)

# initialize params
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

# instantiate optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

# training loop
loss_sequence = []
num_examples = len(Xtrain)
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = log_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    print("Epoch {}, loss: {}".format(e, cumulative_loss))

# plot learning curve
plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)
plt.show()

# calculate accuracy
num_correct = 0.0
num_total = len(Xtest)
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    output = net(data)
    prediction = (nd.sign(output) + 1) / 2
    num_correct += nd.sum(prediction == label)
print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar()/num_total, num_correct.asscalar(), num_total))
