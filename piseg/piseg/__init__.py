# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.polyp_instance_new_baseline_dataset_mapper import POLYPInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.polyp_instance_test_new_baseline_dataset_mapper import POLYPInstanceTestNewBaselineDatasetMapper
from .data.dataset_mappers.polyp_semantic_new_baseline_dataset_mapper import POLYPSemanticNewBaselineDatasetMapper
from .data.dataset_mappers.polyp_semantic_test_new_baseline_dataset_mapper import POLYPSemanticTestNewBaselineDatasetMapper
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer

from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
