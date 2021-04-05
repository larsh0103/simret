import yaml

from utils import SimclrDataSet, get_data_loaders, get_transforms
from simclr import SimCLR
import fire




def train(config_file):
    if not config_file:
        print("Config file must be provided")
    
    ##load config from file
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    
    ## initiate dataset loaders 
    transforms = get_transforms(input_shape=config['dataset']['input_shape'])
    dataset = SimclrDataSet(root=config['image_dir'],transform=transforms)
    train_dataloader, val_dataloader = get_data_loaders(dataset=dataset,val_size=config['dataset']['val_size'],
                                                        num_workers=config['dataset']['num_workers'],batch_size=config['batch_size'])
    simclr = SimCLR(train_dataloader,val_dataloader,config)
    simclr.train()

def main():
    fire.Fire(train)

if __name__ == "__main__":
    main()