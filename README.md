# Tool to generate caffe prototxt for DenseNet

### Introduction
We made a tool to generate caffe prototxt for DenseNet from https://github.com/liuzhuang13/DenseNet.

For details of these networks, please read the original paper:
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)


### How to use this tool
```
1. Specify the network configuration, dense block, trainsition layer, and classification layer
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
2. Call build_model() to generate the deploy and train_val prototxt
    xnet = DenseNet()
    xnet.build_model(DENSE_NET_121_CFG)

3. The generated DenseNet121 can be viewed in (http://ethereon.github.io/netscope/#/gist/6af541e7afd34b505fd1bfae53da7040)
```