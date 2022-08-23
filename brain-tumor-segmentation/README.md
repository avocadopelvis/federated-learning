# BRAIN TUMOR SEGMENTATION

![test_gif_BraTS20_Training_001_flair](https://user-images.githubusercontent.com/92647313/178347082-f4cb5c90-9738-4f7a-be5d-abd6f77c1541.gif)


## DATASET
[BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

### Institutions Involved
- Cancer Imaging Program, NCI, National Institutes of Health (NIH), USA
- Center for Biomedical Image Computing and Analytics (CBICA), SBIA, UPenn, PA, USA
- University of Alabama at Birmingham, AL, USA
- University of Bern, Switzerland
- University of Debrecen, Hungary
- MD Anderson Cancer Center, TX, USA
- Washington University School of Medicine in St. Louis, MO, USA
- Heidelberg University, Germany
- Tata Memorial Centre, Mumbai, India

The sub-regions of tumor considered for evaluation are: 
1) The "enhancing tumor" (ET) 
2) The "tumor core" (TC)
3) The "whole tumor" (WT) 

The provided segmentation labels have values of 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.

## MODELS
- U-net [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597)
- U-net++ [Zhou et al. (2018)](https://arxiv.org/abs/1807.10165)
- Attention U-net [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999)

## TO DO
- [ ] Fix weight scaling

