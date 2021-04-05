import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import os
import cv2


class DataWrapper(Dataset):
    def __init__(self, ann_file : str, img_dir : str, class_names: dict, gpu : bool =True):
        """
        Args:
            df: dataframe containing paths to annotation and image files
            class_names: dict containing with class names  (string) as keys and corresponding integer values

        """
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.class_names = class_names
        self.dataset_dicts = get_dataset_dicts(ann_file=self.ann_file,img_dir=self.img_dir)
        self.gpu =gpu

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        img, target = get_image_and_target(record=self.dataset_dicts[idx],gpu=self.gpu)
        return {"image": img, "target": target}

    def num_classes(self):
        return len(self.class_names)

    def label_to_name(self, label):
        return list(self.class_names.keys())[list(self.class_names.values()).index(label)]


def collater(data):
    images = [d["image"] for d in data]
    targets = [d["target"] for d in data]
    return {"images": images, "targets": targets}

def load_data_pairs(ann_file : str, img_dir : str):
    """
    reads annotation file in json format and returns list of images, targets
    where targets is a list of dict with:
        -boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    """
    with open(ann_file) as f:
        anns = json.load(f)

    filenames = anns["images"]
    print(len(filenames))
    print(len(anns['annotations']))
    ids = [i["id"] for i in anns["images"]]
    images = []
    targets = []
    for v in anns["annotations"]:
        bboxlist = []
        classlist = []

        image_id = v["image_id"]
        # print(f"processing annotation : {v['id']}")
        file_name = filenames[ids.index(image_id)]["file_name"].split("/")[2].split("?")[0]
        filename = os.path.join(img_dir,file_name)
        image_filenames.append()
        image = cv2.imread(filename)

        x1 = min(v['bbox'][0],v['bbox'][2])
        x2 = max(v['bbox'][0],v['bbox'][2])
                
        y1 = min(v['bbox'][1],v['bbox'][3])
        y2 = max(v['bbox'][1],v['bbox'][3])
        bboxlist.append([x1,y1,x2,y2]) 
        classlist.append(int(v['category_id']))
        image, target = format_image_target(image,bboxlist,classlist)
        images.append(torch.tensor(data=image, dtype=torch.float32).to("cuda"))
        targets.append(targets) 
    return images,targets

def get_dataset_dicts(ann_file : str, img_dir : str):

    with open(ann_file) as f:
        anns = json.load(f)

    filenames = anns["images"]
    print(len(filenames))
    print(len(anns['annotations']))
    ids = [i["id"] for i in anns["images"]]
    dataset_dicts = []
    for i in ids:

        record = {}
        objs = []
        # file_name = filenames[ids.index(i)]["file_name"].split("/")[2].split("?")[0]
        file_name = [f for f in filenames[ids.index(i)]["file_name"].split("/") if f.endswith(".jpg")][0]
        filename = os.path.join(img_dir,file_name)
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        for v in anns["annotations"]:
            obj = {}
            if ids.index(i) == v["image_id"]:
                print(f"processing annotation : {v['id']}")

                x1 = int(v['bbox'][0])
                y1 = int(v['bbox'][1])
                x2 = x1 +int(v['bbox'][2]) 
                y2 = y1 +int(v['bbox'][3])
                obj = {
                    "bbox": [x1,y1,x2,y2],
                    "bbox_mode": 'XYXY_ABS',
                    "segmentation": v["segmentation"],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts



def get_image_and_target(record : dict,gpu=True):

    bboxlist = [obj['bbox'] for obj in record['annotations']]
    image = np.array(Image.open(record['file_name']))
    image = np.moveaxis(image, -1, 0)
    classlist = [obj['category_id'] for obj in record['annotations']]

    if gpu:
        return (
            torch.tensor(data=image, dtype=torch.float32).to("cuda"),
            {
                "boxes": torch.tensor(np.asarray(bboxlist), dtype=torch.float32).to("cuda"),
                "labels": torch.tensor(np.asarray(classlist), dtype=torch.int64).to("cuda"),
            },
        )
    else:
        return (
            torch.tensor(data=image, dtype=torch.float32),
            {
                "boxes": torch.tensor(np.asarray(bboxlist), dtype=torch.float32),
                "labels": torch.tensor(np.asarray(classlist), dtype=torch.int64),
            },
        )
