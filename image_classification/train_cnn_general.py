'''
This script trains a CNN for image recognition via transfer learning from MobileNet.

USAGE: ipython train_cnn.py [path to data folder] [model_name] [source_model] [batch_size] [epochs] [learning_rate]
example: ipython train_cnn_general.py ~/projects/images/pools/n10 pools_n10 mobilenet 5 10 .001

to-do:

'''


# Dependencies ----
import os
import sys
import mxnet as mx
from mxnet.image import color_normalize
from mxnet import image
from mxnet import init
from mxnet import gluon
import time
from mxnet import autograd
from mxnet import nd
from mxnet.gluon.data.vision import ImageRecordDataset


# Global constants ----
ctx = mx.cpu()

train_augs = [
    image.ResizeAug(224),
    image.HorizontalFlipAug(0.5),  # flip the image horizontally
    image.BrightnessJitterAug(.3), # randomly change the brightness
    image.HueJitterAug(.1)         # randomly change hue
]
test_augs = [
    image.ResizeAug(224)
]


# Functions ----
def transform(data, label, augs):
    # applies augmentations to img data
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')


def train(net, train, validation, ctx, batch_size=64, epochs=10, learning_rate=0.01, wd=0.001):
    train_data = gluon.data.DataLoader(
        train, batch_size, shuffle=True)
    validation_data = gluon.data.DataLoader(
        validation, batch_size)

    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})

    train_util(net, train_data, validation_data,
               loss, trainer, ctx, epochs, batch_size)


def train_util(net, train_iter, validation_iter, loss_fn, trainer, ctx, epochs, batch_size):
    metric = mx.metric.Accuracy()
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_iter):
            st = time.time()
            # ensure context
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # normalize images
            data = color_normalize(data/255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            #  Keep a moving average of the losses
            metric.update([label], [output])
            names = [metric.get()[0]]
            accs = [metric.get()[1]]
            print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(epoch + 1, i + 1, batch_size/(time.time()-st), metric_str(names, accs)))
            # if i%100 == 0:
                # net.collect_params().save('./checkpoints/%d-%d.params'%(epoch, i))
        train_acc = evaluate_accuracy(train_iter, net)
        validation_acc = evaluate_accuracy(validation_iter, net)
        print("Epoch %s | training_acc %s | val_acc %s " % (epoch + 1, train_acc, validation_acc))


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        data = color_normalize(data/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        output = net(data)
        prediction = nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]


def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])


if __name__ == '__main__':
    # arguments
    data_dir = sys.argv[1]
    model_name = sys.argv[2]
    source_model = sys.argv[3]
    batch_size = int(sys.argv[4])
    epochs = int(sys.argv[5])
    lr = float(sys.argv[6])

    # instatiate source and target models
    if source_model == 'mobilenet':
        from mxnet.gluon.model_zoo.vision import mobilenet1_0
        pretrained_net = mobilenet1_0(pretrained=True, prefix='model_')
        net = mobilenet1_0(classes=2, prefix='model_')
    elif source_model == 'vgg19':
        from mxnet.gluon.model_zoo.vision import vgg19_bn
        pretrained_net = vgg19_bn(pretrained=True, prefix='model_')
        net = vgg19_bn(classes=2, prefix='model_')
    elif source_model == 'resnet101':
        from mxnet.gluon.model_zoo.vision import resnet101_v2
        pretrained_net = resnet101_v2(pretrained=True, prefix='model_')
        net = resnet101_v2(classes=2, prefix='model_')
    elif source_model == 'inceptionv3':
        from mxnet.gluon.model_zoo.vision import inception_v3
        pretrained_net = inception_v3(pretrained=True, prefix='model_')
        net = inception_v3(classes=2, prefix='model_')
    elif source_model == 'squeezenet':
        from mxnet.gluon.model_zoo.vision import squeezenet1_0
        pretrained_net = squeezenet1_0(pretrained=True, prefix='model_')
        net = squeezenet1_0(classes=2, prefix='model_')
    else:
        print("Available source models (ordered small -> large):\n['squeezenet', 'mobilenet', 'inceptionv3', 'resnet101', 'vgg19']")

    # pathing
    train_rec = os.path.join(data_dir, 'img.rec')
    # validation_rec = os.path.join(data_dir, 'validation/img.rec')
    validation_rec = '/home/ubuntu/projects/computer_vision/pools/test/img.rec'
    model_out_path = os.path.join(data_dir, '../models/{}_{}_b{}_e{}.params'.format(model_name, source_model, batch_size, epochs))

    # load data
    train_iterator = ImageRecordDataset(
        filename=train_rec,
        transform=lambda X, y: transform(X, y, train_augs)
    )
    validation_iterator = ImageRecordDataset(
        filename=validation_rec,
        transform=lambda X, y: transform(X, y, test_augs)
    )

    # instantiate source model
    # pretrained_net = mobilenet1_0(pretrained=True, prefix='model_')
    pretrained_net = vgg19_bn(pretrained=True, prefix='model_')
    # instantiate target model
    # net = mobilenet1_0(classes=2, prefix='model_')
    net = vgg19_bn(classes=2, prefix='model_')
    # transfer non output layers from source model to target model
    net.features = pretrained_net.features
    # initializing parameters for output layers of target model
    net.output.initialize(init.Xavier())

    # train target model
    start = time.time()
    train(net, train_iterator, validation_iterator, ctx, batch_size=batch_size, epochs=epochs, learning_rate=lr)
    end = time.time()
    print('\nTOTAL TIME ELAPSED: {}s\n'.format(end - start))

    # save target model's learned parameters
    net.collect_params().save(model_out_path)
