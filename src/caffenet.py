
from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
import numpy as np

# helper function for common structures
def bn_scale_relu_conv(bottom, ks, knum, stride=1, pad=0):
    _xbn   = L.BatchNorm(bottom, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    _xscale = L.Scale(_xbn, in_place=True, bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    _xrelu = L.ReLU(_xscale, in_place=True)
    _xconv  = L.Convolution(_xrelu, kernel_size=ks, stride=stride, weight_filler=dict(type='xavier'),
                            bias_term=False,bias_filler=dict(type='constant'), num_output=knum, pad=pad)
    return _xconv


def conv_bn_scale_relu(bottom, ks, knum, stride=1, pad=0):
    _xconv = L.Convolution(bottom, kernel_size=ks, num_output=knum, stride=stride, weight_filler=dict(type='xavier'),
                           bias_term=False, bias_filler=dict(type='constant'), pad=pad)
    _xbn   = L.BatchNorm(_xconv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    _xscale = L.Scale(_xbn, in_place=True,  bias_term=True, filler=dict(value=1), bias_filler=dict(value=0))
    _xrelu = L.ReLU(_xscale, in_place=True)

    return _xrelu


def basic_conv_block(bottom, knum, postfix):
    '''
    basic build block for densenet,128 1x1 convolution, followed by 32 3x3 convolution
    output: concated featuremap
    '''
    #1x1 conv
    _x1x1conv = bn_scale_relu_conv(bottom, 1, knum)
    #3x3 conv
    _x3x3conv = bn_scale_relu_conv(_x1x1conv, 3, 32, pad=1)
    #concat
    _xConcat  = L.Concat(bottom, _x3x3conv, name='concat_'+postfix)
    return _xConcat

def dense_block(net, bottom, knum, blockid, layernum):
    _xinput = bottom
    for i in range(layernum):
        postfix = '{0}_{1}'.format(blockid, i)
        _xConcat = basic_conv_block(_xinput, knum, postfix)
        _xinput = _xConcat

    return _xConcat


def transition_layer(bottom, num_channels):
    # Transition layer: 1x1 conv + average pooling
    _x1x1conv = bn_scale_relu_conv(bottom, 1, num_channels)
    _xpool    = L.Pooling(_x1x1conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    return _xpool

def classfication_layer(bottom, num_ouput):
    # Classification layer: 7x7 global average pool + 1000 InnerProduct + Softmax
    _xpool = L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)
    _fc    = L.InnerProduct(_xpool, num_output= num_ouput ,in_place=False,
                            weight_filler=dict(type='xavier'), bias_term=False,
                            bias_filler=dict(type='constant'))
    return _fc


class DenseNet(object):
    def __init__(self ):

        self.img_rows = 224
        self.img_cols = 224
        self.img_channels = 3

    def build_model(self, net_cfg):
        # Fixme: the generated prototxt does not include val , need manual edit so far
        with open('train_val.prototxt', 'w') as f:
            print(self.construct_model(net_cfg, '/path/to/caffe-train-lmdb', train_val=True, deploy=False), file=f)

        with open('deploy.prototxt', 'w') as f:
            print(self.construct_model(net_cfg, '/path/to/caffe-train-lmdb', train_val=False, deploy=True), file=f)


    def construct_model(self, net_cfg,
                        lmdb, batch_size=256, train_val=False, deploy=False):
        net = caffe.NetSpec()
        if train_val:
            net.data, net.label = L.Data(name='data', source=lmdb, backend=P.Data.LMDB,
                                         batch_size=batch_size, ntop=2,
                                         include=dict(phase=caffe.TRAIN))

        elif deploy:
            net.data = L.Input(name='data', ntop=1,
                               shape=dict(dim=[batch_size, self.img_channels, self.img_rows, self.img_rows]),
                               include=dict(phase=caffe.TEST))

        # 7x7 convolution followed by 3x3 max pooling
        net.conv1 = conv_bn_scale_relu(net.data, 7, 64, stride=2, pad=3)
        net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        # Build the denseblock
        _out = net.pool1
        dense_block_id = 0
        for i, item in enumerate(net_cfg):
            layer_type = item['Type']
            if layer_type == 'DenseBlock':
                _out = dense_block(net, _out, item['KernelNum'], dense_block_id, item['LayerNum'])
                dense_block_id +=1
            elif layer_type == 'Transition':
                _out = transition_layer(_out, item['KernelNum'])
            elif layer_type == 'Classification':
                _out = classfication_layer(_out, item['OutputNum'])
            else:
                raise ValueError, 'layer_type not supported'+item['Type']


        #Connect full connected network to softmax
        net.fc = _out
        if train_val:
            net.loss = L.SoftmaxWithLoss(net.fc, net.label)
            net.acc = L.Accuracy(net.fc, net.label, include=dict(phase=caffe.TEST))
        if deploy:
            net.prob = L.Softmax(net.fc, name='prob')

        return net.to_proto()


DENSE_NET_121_CFG = [
    {'Type': 'DenseBlock',  'LayerNum':6, 'KernelNum':128},
    {'Type': 'Transition',  'KernelNum': 128},
    {'Type': 'DenseBlock', 'LayerNum': 12, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 256},
    {'Type': 'DenseBlock', 'LayerNum': 24, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 512},
    {'Type': 'DenseBlock', 'LayerNum': 16, 'KernelNum': 128},
    {'Type': 'Classification', 'OutputNum':1000},
]

DENSE_NET_169_CFG = [
    {'Type': 'DenseBlock',  'LayerNum':6, 'KernelNum':128},
    {'Type': 'Transition',  'KernelNum': 128},
    {'Type': 'DenseBlock', 'LayerNum': 12, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 256},
    {'Type': 'DenseBlock', 'LayerNum': 32, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 512},
    {'Type': 'DenseBlock', 'LayerNum': 32, 'KernelNum': 128},
    {'Type': 'Classification', 'OutputNum':1000},
]

DENSE_NET_201_CFG = [
    {'Type': 'DenseBlock',  'LayerNum':6, 'KernelNum':128},
    {'Type': 'Transition',  'KernelNum': 128},
    {'Type': 'DenseBlock', 'LayerNum': 12, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 256},
    {'Type': 'DenseBlock', 'LayerNum': 48, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 512},
    {'Type': 'DenseBlock', 'LayerNum': 32, 'KernelNum': 128},
    {'Type': 'Classification', 'OutputNum':1000},
]

DENSE_NET_264_CFG = [
    {'Type': 'DenseBlock',  'LayerNum':6, 'KernelNum':128},
    {'Type': 'Transition',  'KernelNum': 128},
    {'Type': 'DenseBlock', 'LayerNum': 12, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 256},
    {'Type': 'DenseBlock', 'LayerNum': 64, 'KernelNum': 128},
    {'Type': 'Transition',  'KernelNum': 512},
    {'Type': 'DenseBlock', 'LayerNum': 48, 'KernelNum': 128},
    {'Type': 'Classification', 'OutputNum':1000},
]


if __name__ == '__main__':
    xnet = DenseNet()
    xnet.build_model(DENSE_NET_201_CFG)
