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

1. Download the Kvasir-SEG dataset from https://datasets.simula.no/kvasir-seg/ and unzip the image and image-mask folder into the datasets/images/ folder
2. The data annotation for polyp instance segmentation can be found at ./datasets/ folder

##  Usage

####  1. Training

```bash
!python "./piseg/train_net.py" --config-file "$config-file" --num-gpus 1 --resume DATASETS.TRAIN '("kvasir_instance_train",)' DATASETS.TEST '("kvasir_instance_val",)' DATALOADER.NUM_WORKERS 12 SOLVER.IMS_PER_BATCH 10 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 7700 SOLVER.STEPS "(5600,7000)" SOLVER.CHECKPOINT_PERIOD 70 TEST.EVAL_PERIOD 70 OUTPUT_DIR "$output_dir"
```
* $config-file: the path to the config file, polyp instance segmentation(./piseg/configs/polyp/instance-segmentation/piseg_R50_bs16_50ep.yaml).
* $output_dir: specify the path to save the checkpoint during the training process.

####  2. Inference

```bash
!python "./piseg/train_net.py" --config-file "./piseg/configs/polyp/instance-segmentation/piseg_R50_bs16_50ep.yaml" --num-gpus 1 --eval-only  DATASETS.TEST '("fold_0_kvasir_instance_test",)'  SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS "./output_piseg_resnet50/model_0007069.pth" OUTPUT_DIR "./output_piseg_resnet50/"
```
* $config-file: the path to the config file, polyp instance segmentation(./piseg/configs/polyp/instance-segmentation/piseg_R50_bs16_50ep.yaml).
* $output_dir: specify the path to save the results during the evaluating process.

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
