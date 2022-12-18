# CLPN
# PyTorch code for the paper "Conditional Laplacian Pyramid Networks for Exposure Correction"

## Abstract

Improper exposures greatly degenerate the visual quality of images. It is challenging to correct various exposure errors in a unified framework as it requires simultaneously handling global attributes and local details under different exposure conditions. Most prior works mainly focus on underexposed scenes, which only cover a relatively small fraction of the possible exposures. In this paper, we propose a conditional Laplacian pyramid network (CLPN) for correcting different types of exposure errors in the same framework. It applies Laplacian pyramid to decompose an improperly exposed image into a low-frequency (LF) component and several high-frequency (HF) components, and then enhances the decomposed components in a coarse-to-fine manner. To consistently correct a wide range of exposure errors, a conditional feature extractor (CFE) is designed to extract the conditional feature from the given image. Afterwards, the conditional feature is used to guide the refinement of LF features, so that a precisely correction for illumination, contrast and color tone can be obtained.
As different frequency components exhibit pixel-wise correlations, the LF components before and after enhancement, together with the coarsest HF component are used to guide the reconstruction of the HF components in upper pyramid layers. By doing so, small structure and fine details can be effectively restored, while noises and artifacts can be well suppressed. Extensive experiments demonstrate that the proposed CLPN is more effective than state-of-the-art methods on correcting various exposure conditions ranging from serious underexposure to dramatical overexposure.


## Dependencies
* python==3.7.5
* torch==1.7.1
* torchvision==0.8.2
* tensorboard==2.5.0
* numpy==1.19.5
* opencv-python==4.2.0.34
  


## Test and Train 

The source code of CLPN will be available after the publication of the paper.
