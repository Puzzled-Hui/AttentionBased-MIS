# AttentionBased-MIS
Medical Image Segmentation Based on Attention mechanism(Fine-Tune for natural image[semantic/instance] segmentation).

This project is dedicated to 
- Collecting and re-implementing different attention mechanisms,transforming them modular and portable.
- Proposing  novel attention mechanisms tailed for 3D data Segmentation.

Main purpose is used in 3D Medical Image Segmentation.Fine-tune for Other CV task need attention easily meanwhile.

## data
Base dataset and derived dataset...[Coming Soon]()

## models
This section include basic model(for segmentation or feature extraction) and different attention mechanisms.Each attention mechanism can recalibrate multi-dim feature maps across their own functional domain.

Most attention mechanisms can be modularized and integrated into sub feature maps(e.g. each encoder in UNet or each block in ResNet) 
if not special illustrated. so `models` has basic models and attention modules. You can use this command to  combine your own concrete model, for example:

```
--basic model ResNet --attention module SE 
--basic model 3D UNet --attention module AG
... ...
```

You can Specific where the attention modules inserted in, **Default** is after each block/encoder/decoder.

### Basic model
**3D UNet**: [paper](https://arxiv.org/pdf/1606.06650.pdf)|[reimplemented code: Coming Soon]()








