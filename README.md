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

Most attention mechanisms can be modularized and integrated into any sub feature maps(e.g. each encoder in UNet or each block in ResNet) 
if not special noted. so `models` has basic models and attention modules. You can combine **Model Name** with **Attention Module Name** to
construct your own concrete model, for example:

```
--basic model ResNet --attention module CBAM 
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


### Attention module
**Class Activation Map**: [paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `CAM` [notes: Coming soon]()

**Spatial Transformer Net**: [paper](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `STN` [notes: Coming soon]()

**Squeeze-and-Excitation**: [paper](http://www.robots.ox.ac.uk:5000/~vgg/publications/2018/Hu18/hu18.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `SE` [notes: Coming soon]()

**CBAM**: [paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `CBAM` [notes: Coming soon]()

**Dual Attention**: [paper](https://www.zpascal.net/cvpr2019/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `DN` [notes: Coming soon]()

**Project&Excitation**: [paper](https://arxiv.org/pdf/1906.04649.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `PE` [notes: Coming soon]()

**Attention U-Net**: [paper](https://arxiv.org/pdf/1804.03999.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `AG` [notes: Coming soon]()

**Volumetric Attention**: [paper](http://www.svcl.ucsd.edu/people/xdwang/MICCAI_2019.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `VA` [notes: Coming soon]()

**Feature Correlation Attention**: [paper](https://arxiv.org/pdf/1906.02999.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `FCA` [notes: Coming soon]()

**Hierarchical Attention Net**: [paper](https://arxiv.org/pdf/1911.08777.pdf)|[reimplemented code: Coming Soon]()      Attention Module Name: `HAN` [notes: Coming soon]()

**Ours**: [paper: Coming Soon]()|[Source code: Coming Soon]()      Attention Module Name: `***` [notes: Coming soon]()

Above models and attention modules have been experimented and still many other models waiting for test. 
This project will be update consistently and Welcome to advise good base model or attention modules.
 


