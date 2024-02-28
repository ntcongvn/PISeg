import time

import numpy as np
import torch
import sys
from torch import nn
from torch.nn import functional as F
#from torchvision.transforms.functional import center_crop
from torchvision.transforms import RandomCrop
from copy import deepcopy
from .rpn_texture_synthesis import ExpandTexture, SaveImageFromNumpy
from .texture_extract_backbone import build_texture_extract_backbone

from detectron2.structures import ImageList


class TextureExtractor(nn.Module):
    def __init__(self,cfg):
      super().__init__()
      
      self.out_channel=cfg.MODEL.TEXTURE_EXTRACTOR.OUT_CHANNEL
      #self.randomcrop_size=cfg.MODEL.TEXTURE_EXTRACTOR.RANDOMCROP_SIZE                    #768
      self.out_features= cfg.MODEL.TEXTURE_EXTRACTOR.OUT_FEATURES                         #["t_res5"] # ["t_res2", "t_res3", "t_res4", "t_res5"]
      #self.texture_filter_pairs=cfg.MODEL.TEXTURE_EXTRACTOR.TEXTURE_FILTER_PAIRS          
      #if self.randomcrop_size is not None:
      #  self.random_crop=RandomCrop(size=self.randomcrop_size,pad_if_needed=True,fill=0)
      #  self.size_in=self.randomcrop_size
      #else:
      self.size_in=cfg.INPUT.IMAGE_SIZE
      self.size_divisibility=cfg.MODEL.TEXTURE_EXTRACTOR.SIZE_DIVISIBILITY  #16 
      self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
      self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
      
      #We use resnet18 to synthesis texture feature
      self.backbone=build_texture_extract_backbone(cfg)
      
      proj_list = []
      channels_list=[v.channels for k,v in self.backbone.output_shape().items()]
      # from high resolution to low resolution (t_res2 -> t_res2)
      for in_channels in channels_list:
        proj_list.append(nn.Sequential(
            nn.Conv2d(in_channels, self.out_channel, kernel_size=1),
            nn.GroupNorm(32, self.out_channel),
            nn.AdaptiveAvgPool2d((1, 1))
        ))
      self.proj_list = nn.ModuleList(proj_list)
      for proj in self.proj_list:
        nn.init.xavier_uniform_(proj[0].weight, gain=1)
        nn.init.constant_(proj[0].bias, 0)

    @property
    def device(self):
        return self.pixel_mean.device
   

    #def remove_texture(self,features,texture_features):
    #  for pair in self.texture_filter_pairs:
    #    features[pair[0]]= features[pair[0]]+ (features[pair[0]]-texture_features[pair[1]])
    #  return features

    def forward(self, images):
      images = [(x - self.pixel_mean) / self.pixel_std for x in images]
      images = ImageList.from_tensors(images, self.size_divisibility)
      texture_features=self.backbone(images.tensor)
      for idx, stage in enumerate(self.out_features):
        texture_features[stage]=self.proj_list[idx](texture_features[stage])
      return texture_features   










 