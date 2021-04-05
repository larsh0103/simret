import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image

from simret import backbone_util, retinanet
from simret.anchor_utils import AnchorGenerator
import cv2

def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def draw_rectangle(image,box):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)


def load_model(pth_file: str, backbone_name: str = "resnet18") -> nn.Module:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    backbone = backbone_util.resnet_fpn_backbone(backbone_name=backbone_name, pretrained=False, trainable_layers=5)
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(
        sizes=tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model: nn.Module = retinanet.RetinaNet(backbone, num_classes=2, anchor_generator=anchor_generator)
    checkpoint = torch.load(pth_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def detect(
    image, model: nn.Module, device: str = "cpu", threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # image = np.moveaxis(np.array(Image.open(img_path)),-1,0)
    # image = np.moveaxis(image,-1,0)
    input = [torch.tensor(data=image, dtype=torch.float32).to(device)]
    model.to(device)
    model.eval()

    start = time.perf_counter()
    with torch.no_grad():
        results = model(input)
    elapsed = time.perf_counter() - start
    logger.debug(f"Pure detection time: {elapsed} sec")
    assert len(results) == 1

    results = results[0]
    scores = results["scores"]
    boxes = results["boxes"]
    labels = results["labels"]

    idxs: torch.Tensor = torch.where(scores > threshold)[0]
    return scores[idxs], boxes[idxs], labels[idxs]

