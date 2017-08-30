

from __future__ import print_function
import os, sys
sys.path.append(os.path.abspath("../../src"))

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

class CaffeUnetClf(object):
    def __init__(self, gpuid):
        self._gpuid  = gpuid
        self._deploy = 'deploy.prototxt'
        self._net    = None

        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1


    def build_model(self):
        #Fixme: the generated prototxt does not include val , need manual edit so far
        with open('train_val.prototxt', 'w') as f:
            print(self.caffenet('/path/to/caffe-train-lmdb', train_val=True), file=f)

        with open('deploy.prototxt', 'w') as f:
            print(self.caffenet('/path/to/caffe-val-lmdb', batch_size=32, deploy=True), file=f)

    def caffenet(self, lmdb, batch_size=256, train_val=False, deploy=False):
        net = caffe.NetSpec()
        if train_val:
            net.data, net.label = L.Data(name='data', source=lmdb, backend=P.Data.LMDB,
                                 batch_size=batch_size, ntop=2,
                                 include=dict(phase=caffe.TRAIN))

        elif deploy:
            net.data = L.Input(name='data', ntop=1,
                           shape=dict(dim=[batch_size, 1, 32, 32]),
                           include=dict(phase=caffe.TEST))

        net.conv1, net.relu1 = conv_relu(net.data, 3, 32, stride=1, pad=1)
        net.conv1x, net.relu1x = conv_relu(net.relu1, 3, 32, stride=2, pad=1)

        net.conv2, net.relu2 = conv_relu(net.relu1x, 3, 64, stride=1, pad=1)
        net.conv2x, net.relu2x = conv_relu(net.relu2, 3, 64, stride=2, pad=1)

        net.conv3, net.relu3 = conv_relu(net.relu2x, 3, 128, stride=1, pad=1)
        net.conv3x, net.relu3x = conv_relu(net.relu3, 3, 128, stride=2, pad=1)

        net.conv4, net.relu4 = conv_relu(net.relu3x, 3, 256, stride=1, pad=1)
        net.conv4x, net.relu4x = conv_relu(net.relu4, 3, 256, stride=2, pad=1)

        net.conv5, net.relu5 = conv_relu(net.relu4x, 3, 512, stride=1, pad=1)

        net.fc6 = L.InnerProduct(net.relu5, num_output=2)

        if train_val:
            net.loss = L.SoftmaxWithLoss(net.fc6, net.label)
            net.acc = L.Accuracy(net.fc6, net.label, include=dict(phase=caffe.TEST))
        if deploy:
            net.prob = L.Softmax(net.fc6, name='prob')
        return net.to_proto()

    def load_weights_from_file(self, snapshotfile=None):
        caffe.set_mode_gpu()
        caffe.set_device(self._gpuid)
        self._net = caffe.Net(self._deploy, caffe.TEST, weights=snapshotfile)
        self.summary()

    def summary(self):
        for name, blob in self._net.blobs.iteritems():
            print (name + '\t' + str(blob.data.shape))

    def predict_by_batch(self, xcroplst, batchsize=32):
        out = np.ndarray((len(xcroplst), 2), dtype=float)
        self._net.blobs['data'].reshape(batchsize, self.img_channels, self.img_rows, self.img_cols)
        for batchstart in range(0, len(xcroplst)/batchsize*batchsize, batchsize):
            for i in range(batchsize):
                xcrop = xcroplst[batchstart+i]
                self._net.blobs['data'].data[i] = xcrop.reshape((1,  self.img_channels, self.img_rows, self.img_cols))
            probs = self._net.forward()
            out[batchstart:batchstart+batchsize] = probs['prob']

        #for last batch
        last_batch_start = len(xcroplst)/batchsize*batchsize
        self._net.blobs['data'].reshape(len(xcroplst) - last_batch_start, self.img_channels, self.img_rows, self.img_cols)
        for i in range(0, len(xcroplst) - last_batch_start , 1):
            xcrop = xcroplst[last_batch_start + i]
            self._net.blobs['data'].data[i] = xcrop.reshape((1, self.img_channels, self.img_rows, self.img_cols))
        probs = self._net.forward()
        out[last_batch_start:len(xcroplst)] = probs['prob']

        return out




def scan_best_model():
    from resnet9_dark import test_single_image_range
    imgfile = "/home/victor/victor_xcos/xcos/images/ww16.2/Pattern Tray 01/Images/C0000002_BCB1MCF2_BBI SI-BLand-Indentation.jpg"
    for modelfile in os.listdir("../../snapshot"):
        if modelfile.endswith('caffemodel.h5'):
            xmodel = os.path.join("../../snapshot", modelfile)
            xnet = CaffeUnetClf(1)
            xnet.load_weights_from_file(xmodel)
            test_single_image_range(xnet,
                                    imgfile,
                                    os.path.join("../../snapshot", modelfile+"ww16.2_tray1_02.png"),
                                    heatmapfile=os.path.join("../../snapshot", modelfile + "ww16.2_tray1_02_heatmap.png"),
                                    stride=4)
    print ("done")

def inference_image(xnet, infile, outfile, batchsize, stride=8):
    import time
    import cv2
    from cpu import CPU
    from data_light_generator import get_crops_among_rects

    xcpu = CPU(infile)
    _, outboxrange, innerbox = CPU(infile).get_roi_areas()

    croplst = get_crops_among_rects(infile, outboxrange, innerbox, 32, stride)

    ximg = cv2.imread(infile)

    xlist = [xcrop for xcrop, bbox in croplst]

    tstart = time.time()
    out = xnet.predict_by_batch(xlist)
    tend  = time.time()

    mlist = list()
    for i, (xcrop, bbox) in enumerate(croplst):
        pred = out[i,1]
        mlist.append((bbox, pred))
        if pred > 0.5:
            x, y, w, h = bbox
            cv2.rectangle(ximg, (x, y), (x + w, y + h), (0, 0, 255))

    cv2.imwrite(outfile, ximg)
    print ("process ", len(xlist), " crops takes  ", tend-tstart, " @ batchsize ", batchsize)


def inference_benchmark():
    imgfile = "/home/victor/victor_xcos/xcos/images/ww16.2/Pattern Tray 01/Images/C0000002_BCB1MCF2_BBI SI-BLand-Indentation.jpg"
    modelfile = ''
    xmodel = os.path.join("../../snapshot", modelfile)
    xnet = CaffeUnetClf(1)
    xnet.load_weights_from_file(xmodel)
    for batchsize in [32, 64, 128, 256, 512, 1024]:
        inference_image(xnet,
                        imgfile,
                        os.path.join("../../snapshot", modelfile+"ww16.2_tray1_02.png"),
                        batchsize,
                        stride=4)

if __name__ == '__main__':
    xnet = CaffeUnetClf(1)
    xnet.build_model()
    #xnet.summary()

    #xnet.build_model()
    #scan_best_model()
    #inference_benchmark()