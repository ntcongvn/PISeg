# PISeg
PISeg: Polyp Instance Segmentation With Texture Denoising And Adaptive Region

##  Introduction

This repository contains the PyTorch implementation of PISeg, Polyp Instance Segmentation With Texture Denoising And Adaptive Region. 
Our proposed PISeg has three modules: a Backbone, a Pixel Decoder, and a Transformer Decoder, as shown in the figure below. First, the backbone handles feature extraction of the input endoscopic image. Besides, the texture image is also extracted by a Texture extractor and used to reduce texture noise in the image feature. Next, the Pixel Decoder upsamples image features into high-resolution features, which are then used by the Transformer Decoder module (TDM) to query polyp instances. A set of Adaptive region queries is responsible for detecting regions containing polyp signs from received feature maps. These region queries are learned through Transformer decoder layers (TDL) and are guaranteed to be independent of each other by a Region loss function. Besides, a set of object queries will be synthesized based on adaptive regions to generate object embeddings for classifying and segmenting polyps.

![model](figures/PISEG_Overview.jpg)

##  Install dependencies

Dependent libraries
* torch
* torchvision 
* opencv
* ninja
* fvcore
* iopath

Install detectron2 and PISeg.

```bask
# Under your working directory
# Install Detectron2
cd ./detectron2
!python setup.py build develop
cd ..

#Install requirements for piseg
cd ./piseg
!pip install -r requirements.txt
cd ..

cd ./piseg/piseg/modeling/pixel_decoder/ops
!sh make.sh


```

##  Polyp instance segmentation dataset
[Kvasir-SEG](<https://datasets.simula.no/kvasir-seg/>) is the most commonly used benchmark of polyps segmentation captured from real-world environments. However, the dataset and others only provide binary semantic masks labeling image regions as polyp or non-polyp. In this work, we take further steps to propose an annotated dataset for pedunculated and sessile polyp instance segmentation. Specifically, we leverage 1,000 polyp images from the Kvasir-SEG dataset, separate the provided polyp semantics into distinguished instances of the two classes, and annotate its masks. 

we hereby walk you through the configuration to generate the preprocessed data for training and evaluation. 
1. Download the LIDC-IDRI Radiologist Annotations/Segmentations (XML format) from https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2 and change the scripts/config.py line 24 'annos_dir' to your downloaded path

2. Download the LIDC-IDRI CT images and the corresponding lung segmentation mask from LUNA16 challenge https://luna16.grand-challenge.org/Download/

    Move all downloaded CT images from the 10 folders to one folder, and change the scripts/config.py line 18 'data_dir' to the CT images directory (combining sub folders into one folder) and change the scripts/config.py line 27 'lung_mask_dir' to the segmentation mask.

    Explanantions on some intermediate results saved: ctr_arr_save_dir will be the place to save the parsed intermediate nodule masks for each annotator, and mask_save_dir is the folder to save the merged nodule mask. In mask_save_dir, there will be 1 - 4 four folders, containing nodule masks that are annotated by at least 1 - 4 annotators respectively.

3. XXX


##  Usage

####  1. Training

```bash
bash train_single_fold.sh $fold $checkpoint
```
* $fold: Fold's index is specified to model train. An integer value in the range 0 to 5.
* $checkpoint: is optional, specify the path to the checkpoint to resume training the model.


####  2. Inference

```bash
python test.py --mode "eval" --test-set-name $testsetname --weight $weight --out-dir $outdir
```
* $testsetname: Path to the csv file containing patient ids. There are 6 csv test files corresponding for 6 folds in scripts/split/cross_val/ for a six-fold cross-validation process
* $weight: Path to weight file.
* $outdir: Path to directory where inference results will be stored.

You will see the results of FROC analysis both saved to files and printed on the screen.


##  Acknowledgement

Part of the code was adpated from [Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation](<https://github.com/facebookresearch/Mask2Former>)

```bash
@INPROCEEDINGS{9878483,
  author={Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and Kirillov, Alexander and Girdhar, Rohit},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Masked-attention Mask Transformer for Universal Image Segmentation}, 
  year={2022},
  volume={},
  number={},
  pages={1280-1289},
  keywords={Image segmentation;Shape;Computational modeling;Semantics;Computer architecture;Transformers;Feature extraction;Segmentation;grouping and shape analysis; Recognition: detection;categorization;retrieval},
  doi={10.1109/CVPR52688.2022.00135}
}
```
