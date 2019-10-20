# Pytorch Segmentation 

PyTorch Implementation segmentation models

Segmentation models

- UNet : [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- SegNet : [A Deep Convolutional Encoder-Decoder
Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)
- DenseNet : [The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)

## Usage

simple example

```
python train.py --model UNet
```

You can also use multiple GPU to train models.

```
python train.py --model UNet --device1 0 --device2 1 --device3 2
```

You can see more detailed arguments.

```
python train.py -h
```
