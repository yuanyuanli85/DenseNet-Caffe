
from __future__ import print_function

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
import numpy as np

# helper function for common structures

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant'), num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'),
                        num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)



def bn_scale_relu_conv(bottom, ks, knum, stride=1, pad=0):
    _xbn   = L.BatchNorm(bottom)
    _xscale = L.Scale(_xbn, in_place=True)
    _xrelu = L.ReLU(_xscale, in_place=True)
    _xconv  = L.Convolution(_xrelu, kernel_size=ks, stride=stride, weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant'), num_output=knum, pad=pad)
    return _xconv


def conv_bn_scale_relu(bottom, ks, knum, stride=1, pad=0):
    _xconv = L.Convolution(bottom, kernel_size=ks, stride=stride, weight_filler=dict(type='xavier'),
                           bias_filler=dict(type='constant'), num_output=knum, pad=pad)
    _xbn   = L.BatchNorm(_xconv)
    _xscale = L.Scale(_xbn, in_place=True)
    _xrelu = L.ReLU(_xscale, in_place=True)

    return _xrelu


def basic_conv_block(bottom, postfix):
    '''
    basic build block for densenet,128 1x1 convolution, followed by 32 3x3 convolution
    output: concated featuremap with prior layer's featuremap
    '''
    #1x1 conv
    _x1x1conv = bn_scale_relu_conv(bottom, 1, 128)
    #3x3 conv
    _x3x3conv = bn_scale_relu_conv(_x1x1conv, 3, 32, pad=1)
    #concat
    _xConcat  = L.Concat(bottom, _x3x3conv, name='concat_'+postfix)
    return _xConcat

def dense_block(net, bottom, blockid, layernum):
    _xinput = bottom
    for i in range(layernum):
        postfix = '{0}_{1}'.format(blockid, i)
        _xConcat = basic_conv_block(_xinput, postfix)
        _xinput = _xConcat

    return _xConcat


def transition_layer(bottom):
    # Transition layer: 1x1 conv + average pooling
    _x1x1conv = bn_scale_relu_conv(bottom, 1, 128)
    _xpool    = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return _xpool

def classfication_layer(bottom):
    # Classification layer: 7x7 global average pool + 1000 InnerProduct + Softmax
    _xpool = L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    pass

class DenseNet(object):
    def __init__(self, gpuid):
        self._gpuid  = gpuid
        self._deploy = 'deploy.prototxt'
        self._net    = None

        self.img_rows = 224
        self.img_cols = 224
        self.img_channels = 3

    def build_model(self):
        # Fixme: the generated prototxt does not include val , need manual edit so far
        with open('deploy.prototxt', 'w') as f:
            print(self.caffenet('/path/to/caffe-train-lmdb', train_val=False, deploy=True), file=f)


    def caffenet(self, lmdb, batch_size=256, train_val=False, deploy=False):
        net = caffe.NetSpec()
        if train_val:
            net.data, net.label = L.Data(name='data', source=lmdb, backend=P.Data.LMDB,
                                 batch_size=batch_size, ntop=2,
                                 include=dict(phase=caffe.TRAIN))

        elif deploy:
            net.data = L.Input(name='data', ntop=1,
                           shape=dict(dim=[batch_size, self.img_channels, self.img_rows, self.img_rows]),
                           include=dict(phase=caffe.TEST))

        #7x7 convolution followed by 3x3 max pooling
        net.conv1 = conv_bn_scale_relu(net.data, 7, 64, stride=2, pad=3)
        net.pool1 = max_pool(net.conv1, 3, stride=2)

        # DenseBlock1 : 6
        net.denseblock1 = dense_block(net, net.pool1, 1, 6)
        net.transition1 = transition_layer(net.denseblock1)

        # DenseBlock2: 12
        net.denseblock2 = dense_block(net, net.transition1, 2, 12)
        net.transition2 = transition_layer(net.denseblock2)

        # DenseBlock2: 24
        net.denseblock3 = dense_block(net, net.transition2, 3, 24)
        net.transition3 = transition_layer(net.denseblock3)

        # DenseBlock2: 16
        net.denseblock4 = dense_block(net, net.transition3, 3, 16)

        return net.to_proto()



if __name__ == '__main__':
    xnet = DenseNet(0)
    xnet.build_model()
