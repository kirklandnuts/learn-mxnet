'''
This script classifies images (from urls) using a model with loaded parameters.

USAGE: ipython classify_image.py [path to saved parameters] [label1 value] [label0 value]
example: ipython classify_image.py ~/projects/images/pools/models/pools_n10.params 'pool' 'no pool'
'''


# Dependencies
import mxnet as mx
from mxnet import image
from mxnet.image import color_normalize
import matplotlib.pyplot as plt
from mxnet.gluon.model_zoo.vision import mobilenet1_0
import requests
from io import BytesIO
from mxnet import nd
import sys


# Global Constants ----
test_augs = [
    image.ResizeAug(224)
]
ctx = mx.cpu()

# Functions ----
def predict(net, url, label1, label0, show_img = False):
    data = read_image(url)
    data = data.expand_dims(axis=0)
    data = color_normalize(data/255,
                           mean=nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                           std=nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
    out = softmax(net(data.as_in_context(mx.cpu())))
    prediction = [label0, label1][int(nd.argmax(out, axis=1).asscalar())]
    print('Probability of {}: {}'.format(label1, out[0][1].asscalar()))
    print('Probability of {}: {}'.format(label0, out[0][0].asscalar()))
    print('That image (probably) contained {}.\n'.format(prediction))
    if show_img:
        plt.imshow(img/255)
        plt.subplot(1, 2, 2)
        plt.show()


def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')


def read_image(url):
    with requests.get(url) as f:
        img = image.imdecode(f.content)
    data, _ = transform(img, -1, test_augs)
    return data


def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms


# Main ----
if __name__ == '__main__':
    # pathing
    params_path = sys.argv[1]
    label_1 = sys.argv[2]
    label_0 = sys.argv[3]

    print('==== Loading model')
    # instantiate model
    net = mobilenet1_0(classes=2, prefix='model_')

    # load saved parameters
    net.collect_params().load(params_path, ctx)
    print('==== Model loaded; ready to classify\n')

    # prediction loop
    url = input('Please enter URL (\'q\' to exit): ')
    while url != 'q':
        predict(net, url, label_1, label_0)
        url = input('Please enter URL (\'q\' to exit): ')
