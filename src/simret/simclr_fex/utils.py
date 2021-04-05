import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from gaussian_blur import GaussianBlur


np.random.seed(0)

class SimclrDataSet(ImageFolder):
    """ImageFolder dataset that returns two tranformed images and class target
    """
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample1, sample2, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample1 = self.transform(sample)
            sample2 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, sample2, target

def get_transforms(input_shape=(256,256,3)):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=input_shape[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            GaussianBlur(kernel_size=int(0.1 * input_shape[0])),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
    )
    return data_transforms

def get_data_loaders(dataset,val_size=0.05,num_workers=2,batch_size=64):
    # obtain training indices that will be used for validation
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(val_size * num_train))
    train_idx, val_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, drop_last=True,
    )
    return train_loader, val_loader
