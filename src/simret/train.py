import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.tensorboard import SummaryWriter

from retinanet import backbone_util, retinanet,evaluation
from retinanet.anchor_utils import AnchorGenerator
from retinanet.dataloader import DataWrapper, collater
from simclr_fex.resnet_simclr import ResNetSimCLR


def train_model(
    model,
    img_dir,
    val_img_dir,
    optimizer,
    scheduler,
    ann_file: str,
    val_ann_file: str,
    class_names: dict,
    checkpoints_folder="checkpoints",
    batch_size: int = 3,
    epochs: int = 10,
    use_gpu: bool = True,
):

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    val_writer = SummaryWriter(os.path.join(checkpoints_folder, "validation"))
    train_writer = SummaryWriter(os.path.join(checkpoints_folder, "training"))

    dataset_train = DataWrapper(ann_file=ann_file, img_dir=img_dir, class_names=class_names,gpu=use_gpu)
    dataset_val = DataWrapper(ann_file=val_ann_file, img_dir=val_img_dir, class_names=class_names,gpu=use_gpu)

    sampler_train = RandomSampler(data_source=dataset_train)
    sampler_val = RandomSampler(data_source=dataset_val)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train, collate_fn=collater)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, collate_fn=collater)

    for i in range(epochs):
        classifacation_loss = []
        bbox_regression_loss = []
        epoch_loss = []
        model.train()
        for j, data in enumerate(dataloader_train):

            optimizer.zero_grad()

            losses = model(data["images"], data["targets"])
            #         print(losses['classification'],losses['bbox_regression'])
            loss = losses["classification"] + losses["bbox_regression"]

            epoch_loss.append(loss.item())
            print(
                f"step:{str(j)}, classificaiton_loss:{losses['classification'].item()} bbox_regression:{losses['bbox_regression'].item()}"
            )

            if bool(loss == 0):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            bbox_regression_loss.append(losses["bbox_regression"].item())
            classifacation_loss.append(losses["classification"].item())
        print(f"epoch:{i}, classificaiton_loss:{np.mean(classifacation_loss)} bbox_regression:{np.mean(bbox_regression_loss)}")

        train_writer.add_scalar("reg_loss", np.mean(bbox_regression_loss), global_step=i)
        train_writer.add_scalar("class_loss", np.mean(classifacation_loss), global_step=i)

        # validation and evaluation
        for j, data in enumerate(dataloader_val):
            classifacation_loss = []
            bbox_regression_loss = []
            losses = model(data["images"], data["targets"])
            bbox_regression_loss.append(losses["bbox_regression"].item())
            classifacation_loss.append(losses["classification"].item())

        print(
            f"epoch:{i}, val_classificaiton_loss:{np.mean(classifacation_loss)} val_bbox_regression:{np.mean(bbox_regression_loss)}"
        )
        mAP = evaluation.evaluate(dataset_val, model)

        all_mAP = []
        for clss, score in mAP.items():
            all_mAP.append(score[0])
        all_mAP = np.mean(all_mAP)
        val_writer.add_scalar("reg_loss", np.mean(bbox_regression_loss), global_step=i)
        val_writer.add_scalar("class_loss", np.mean(classifacation_loss), global_step=i)
        val_writer.add_scalar("mAP", np.mean(all_mAP), global_step=i)

        scheduler.step(np.mean(epoch_loss))

        # save the entire model
        model_dict = {
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict(),
            "regression_loss": bbox_regression_loss,
            "classification_loss": classifacation_loss,
        }

        torch.save(model_dict, os.path.join(checkpoints_folder, f"model-e{i:03}.pth"))

    return model_dict


def _initialize_simclr_pretrained_backbone(backbone_name: str = "resnet18", simclr_model_dict: str = None):
    # Initialize blank resnet feature extractor
    backbone = None
    if simclr_model_dict:
        smclr = ResNetSimCLR(base_model=backbone_name, out_dim=256)
        smclr.load_state_dict(torch.load(simclr_model_dict))
        backbone = backbone_util.get_resnet_backbone(backbone_name=backbone_name)
        # Load simclr weights. Need to get rid of two top linear layers in simclr. These are accessory to resnet backbone.
        backbone.features = nn.Sequential(*list(smclr.children())[:-2])

    # add feature pyriamid ontop of feature extractor, so that it outputs the feature maps that retinanet expects.
    backbone = backbone_util.resnet_fpn_backbone(
        backbone_name=backbone_name, pretrained= simclr_model_dict is None, trainable_layers=5, backbone=backbone
    )
    return backbone


def train(
    ann_file: str,
    val_ann_file: str,
    checkpoints_folder: str,
    backbone_name: str = "resnet18",
    img_dir: str = "pokemon/train",
    val_img_dir: str = "pokemon/val",
    simclr_model_dict: str = None,
    class_names: dict = {"GA": 0, "NON_GA": 1},
    epochs: int = 10,
    checkpoint: str = None,
    batch_size=1
):
    """
    Prepares input parameters for training loop, Retinanet model, anchor generator and sets hyperparameters
    --------
    Arguments:
        ann_file : annotation file with which training and evaluation data is loaded.
        backbone_name : string that is the name of a resnet variation, see pytorch documentation to learn which implementations are supported.
        simclr_model_dict : torch model_dict of a resnet model trained using the simclr framework.
        class_names : dict containing class names as keys and corresponding integer values as values.
        epochs : number of training epochs
        checkpoint : path to pretrained model dict

    Returns:
        dictionary containing model state dict and training information.
            model_dict = {
                "epoch": ,
                "model_state_dict": ,
                "optimizer_state_dict": ,
                # "scheduler_state_dict": ,
                "regression_loss": ,
                "classification_loss": ,
            }

    """
    backbone = _initialize_simclr_pretrained_backbone(backbone_name, simclr_model_dict)

    anchor_generator = AnchorGenerator(
        sizes=tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )


    model = retinanet.RetinaNet(backbone, num_classes=2, anchor_generator=anchor_generator)
    if checkpoint:
        try:
            model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
        except Exception:
            print("could not load checkpoint")
            pass
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    use_gpu = torch.cuda.is_available()

    return train_model(model=model, img_dir=img_dir,val_img_dir=val_img_dir,optimizer=optimizer, 
                        scheduler=scheduler, ann_file=ann_file, val_ann_file=val_ann_file,class_names=class_names, 
                        batch_size=1, epochs=epochs,checkpoints_folder=checkpoints_folder, use_gpu=use_gpu)
