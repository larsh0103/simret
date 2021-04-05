import fire
from train import train
import yaml


def run(config_file=None,
    backbone_name="resnet18",
            ann_file='annotations.json',
            val_ann_file='annotations_val.json',
            checkpoints_folder="/checkpoints",
            img_dir="images",
            val_img_dir="images",
            simclr_model_dict=None,
            class_names={'class1':0,'class2':1},
            epochs=1,
            checkpoint=None,
            batch_size=1,
):  
    if config_file:
        args = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    
    else:
        args = dict(
        backbone_name=backbone_name,
        ann_file=ann_file,
        val_ann_file=val_ann_file,
        checkpoints_folder=checkpoints_folder,
        img_dir=img_dir,
        val_img_dir=val_img_dir,
        simclr_model_dict=simclr_model_dict,
        class_names=class_names,
        epochs=epochs,
        checkpoint=checkpoint,
        batch_size=batch_size,
    )

    train(**args)

def main():
    fire.Fire(run)

if __name__ == "__main__":
    main()