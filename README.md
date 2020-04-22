# AttentionBased-MIS
Medical Image Segmentation Based on Attention mechanism(Fine-Tune for natural image[semantic/instance] segmentation).

This project is dedicated to 
- Collecting and re-implementing basic models and different attention mechanisms,transforming them modular and portable.
- Proposing  novel attention mechanisms tailed for 3D data Segmentation.

Main purpose is used in 3D Medical Image Segmentation.Fine-tune for Other CV task need attention easily meanwhile.

## data
Base dataset and derived dataset...[Coming Soon]()

## models
This section include basic model(for segmentation or feature extraction) and different attention mechanisms.Each attention mechanism can recalibrate multi-dim feature maps across their own functional domain.

Most attention mechanisms can be modularized and integrated into sub feature maps(e.g. each encoder in UNet or each block in ResNet) 
if not special illustrated. so `models` has basic models and attention modules. You can use **Model Name** and **Attention Module Name** to
construct your own concrete model, for example:

```
--basic model ResNet --attention module SE 
--basic model 3D UNet --attention module AG
... ...
```

You can Specific where the attention modules inserted in, **Default** is after each block/encoder/decoder.

### Basic model
**3D UNet**: [paper](https://arxiv.org/pdf/1606.06650.pdf)|[reimplemented code: Coming Soon]()      Model Name: `3D UNet`

**VNet**: [paper](https://arxiv.org/pdf/1606.04797.pdf)|[reimplemented code: Coming Soon]()      Model Name: `VNet`

**DeepMedic**: [paper](https://www.sciencedirect.com/science/article/pii/S1361841516301839)|[reimplemented code: Coming Soon]()      Model Name: `DeepMedic`

**H-DenseUNet**: [paper](https://arxiv.org/pdf/1709.07330.pdf)|[reimplemented code: Coming Soon]()      Model Name: `H-DenseUNet`

**VoxResNet**: [paper](https://arxiv.org/pdf/1608.05895.pdf)|[reimplemented code: Coming Soon]()      Model Name: `VoxResNet`

**U-Net**: [paper](https://arxiv.org/pdf/1505.04597.pdf)|[reimplemented code: Coming Soon]()      Model Name: `U-Net`

**ResNet**: [paper](https://arxiv.org/pdf/1512.03385.pdf)|[code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)      Model Name: `ResNet`

**FCN**: [paper](https://arxiv.org/pdf/1411.4038.pdf)|[reimplemented code: Coming Soon]()      Model Name: `FCN`

**DeepLabV3+**: [paper](https://arxiv.org/pdf/1802.02611.pdf)|[reimplemented code: Coming Soon]()      Model Name: `DeepLabV3+`









