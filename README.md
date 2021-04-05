Simret - Simclr pretrained retinanet

This applies the self-supervised pretraining setup described in "A simple framework for contrastive learning of visual representations", by chen et.al, to train a feature extractor which we use as the backbone for object detection. 

Currently this repo only supports resnets as backbones, and retinanet as the object detection network. 

Most of this repo is derived from other open source implementations, for simclr I have relied on Thalles Silva's work  https://github.com/sthalles/SimCLR
