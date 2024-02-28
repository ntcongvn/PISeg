# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager


CVC_ENDOSCENESTILL_SEMSEG_CATEGORIES = [
    {"color": [0, 255, 0], "id": 1, "name": "polyp"},
]


_PREDEFINED_SPLITS_CVC_ENDOSCENESTILL_SEMSEG = {
    "cvc_endoscenestill_semseg_train": (
        #root dir
        "./datasets",
        #image infor
        "cvc_endoscenestill_semseg_train.json",
    ),
    "cvc_endoscenestill_semseg_val": (
        "./datasets",
        "cvc_endoscenestill_semseg_val.json",
    ),
    "cvc_endoscenestill_semseg_test": (
        "./datasets",
        "cvc_endoscenestill_semseg_test.json",
    ),
}


def get_metadata():
    meta = {}

    stuff_classes = [k["name"] for k in CVC_ENDOSCENESTILL_SEMSEG_CATEGORIES]
    stuff_colors = [k["color"] for k in CVC_ENDOSCENESTILL_SEMSEG_CATEGORIES]

    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(CVC_ENDOSCENESTILL_SEMSEG_CATEGORIES):
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i
    
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_cvc_endoscenestill_semseg_json(json_file, image_dir, meta):
   

    with PathManager.open(image_dir+"/"+json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["images"]:
        image_id = int(ann["id"])
        image_file = os.path.join(image_dir, ann["file_name"])
        sem_label_file = os.path.join(image_dir, ann["mask_name"])
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "sem_seg_file_name": sem_label_file,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_cvc_endoscenestill_annos_sem_seg(
    name, metadata, image_root, json_file
):
    semantic_name = name
    DatasetCatalog.register(
        semantic_name,
        lambda: load_cvc_endoscenestill_semseg_json(json_file,image_root,metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        image_root=image_root,
        json_file=json_file,
        evaluator_type="sem_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_cvc_endoscenestill_annos_sem_seg(root):
    for ( prefix, (root_dir, json_file),) in _PREDEFINED_SPLITS_CVC_ENDOSCENESTILL_SEMSEG.items():
        register_cvc_endoscenestill_annos_sem_seg(
            prefix,
            get_metadata(),
            root_dir,
            json_file,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cvc_endoscenestill_annos_sem_seg(_root)
