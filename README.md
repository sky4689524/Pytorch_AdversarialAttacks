# Pytorch Semgentation and Adversarial attacks

This repository implements the segmentation models and segmentation adversarial attacts by pytorch. The main algorithms are referenced from "Generalizability vs. Robustness: Adversarial Examples for Medical Imaging" by Paschali, M., Conjeti, S., Navarro, F., & Navab, N. at MICCAI 2018. 

There are three segmentation models: UNet, SegNet, and DenseNet. Also, there are three different type of dense adversarial generations : Type A(target to be all background), Type B(target to be top 3 frequency labels), Type C(only one random target)


Segmentation models

- UNet : [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- SegNet : [A Deep Convolutional Encoder-Decoder
Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)
- DenseNet : [The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)

Adversarial Attacks for semantic segmentation DNNs.

- Dense Adversarial Generation : [Adversarial examples for semantic segmentation and object detection](https://arxiv.org/pdf/1703.08603.pdf)

## Usage

`train.py` : train segmentation models

`test.py` : test data with trained models

`adversarial.py` :  generate adversarial examples based on segmentation models

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
