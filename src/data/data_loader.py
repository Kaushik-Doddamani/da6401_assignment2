import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path if it isn’t already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.common_utils import set_seeds

def load_inaturalist_train_val(data_dir,
                               val_ratio=0.2,
                               seed=42,
                               augment=False,
                               resize_dim=224):
    """
    Loads iNaturalist data from 'data_dir', does stratified train/val split.
    Optionally apply data augmentations or just a simple Resize+ToTensor.

    :param data_dir:      Path to the folder containing subfolders of images,
                          e.g. ".../inaturalist_12K_extracted/inaturalist_12K/train"
    :param val_ratio:     Fraction of data to reserve for validation (default 0.2)
    :param seed:          Random seed to ensure reproducible splits
    :param augment:       If True, apply creative data augmentations
    :param resize_dim:    The final resize dimension for height & width
                          (ideally a multiple of 32 for this CNN)
    :return:              (train_dataset, val_dataset, class_names)    
    """
    # Define Transformations
    # If you set augment=True, we'll apply some "creative" transformations.
    # Otherwise, we just do a simple resize + ToTensor().
    if augment:
        transform_list = [
            #  (A) Random resizing and cropping
            T.RandomResizedCrop(size=resize_dim),
            
            #  (B) Random flips
            T.RandomHorizontalFlip(p=0.5),
            
            #  (C) Some random rotation
            T.RandomRotation(degrees=30),
            
            #  (D) Color jitter (brightness/contrast/saturation/hue)
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            
            #  (E) Small chance to invert the colors
            T.RandomInvert(p=0.1),
            
            #  (F) Random perspective distortion
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            
            #  (G) Finally, convert to Tensor
            T.ToTensor(),
            
            #  (H) Optionally, random erase part of the image
            T.RandomErasing(p=0.1)
        ]
    else:
        # Minimal transform
        transform_list = [
            T.Resize((resize_dim, resize_dim)),
            T.ToTensor()
        ]

    transform = T.Compose(transform_list)

    # Full dataset
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir,
                                                    transform=transform)
    class_names = full_dataset.classes
    labels = full_dataset.targets
    indices = list(range(len(full_dataset)))

    # Stratified split
    set_seeds(seed)
    train_indices, val_indices = train_test_split(indices,
                                                  test_size=val_ratio,
                                                  stratify=labels,
                                                  random_state=seed)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(full_dataset, val_indices)
    return train_dataset, val_dataset, class_names


class InatDataModule(pl.LightningDataModule):
    """Data module for iNaturalist images. Creates stratified train/val split."""
    def __init__(self,
                 data_dir: str = "../inaturalist_data/nature_12K_extracted/inaturalist_12K/train",
                 val_ratio: float = 0.2,
                 seed: int = 42,
                 augment: bool = False,
                 resize_dim: int = 224,
                 batch_size: int = 32,
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment
        self.resize_dim = resize_dim
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Called by Lightning at the beginning (once per process in DDP).
        set_seeds(self.seed)

        # Transforms
        if self.augment:
            transform_list = [
                #  (A) Random resizing and cropping
                T.RandomResizedCrop(size=self.resize_dim),
                
                #  (B) Random flips
                T.RandomHorizontalFlip(p=0.5),
                
                #  (C) Some random rotation
                T.RandomRotation(degrees=30),
                
                #  (D) Color jitter (brightness/contrast/saturation/hue)
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                
                #  (E) Small chance to invert the colors
                T.RandomInvert(p=0.1),
                
                #  (F) Random perspective distortion
                T.RandomPerspective(distortion_scale=0.2, p=0.5),
                
                #  (G) Finally, convert to Tensor
                T.ToTensor(),
                
                #  (H) Optionally, random erase part of the image
                T.RandomErasing(p=0.1)
            ]
        else:
            transform_list = [
                T.Resize((self.resize_dim, self.resize_dim)),
                T.ToTensor()
            ]
        transform = T.Compose(transform_list)

        # Full dataset
        full_dataset = torchvision.datasets.ImageFolder(root=self.data_dir, transform=transform)
        self.class_names = full_dataset.classes
        num_classes = len(self.class_names)

        # Stratified splitting
        # ImageFolder stores labels in `full_dataset.targets`
        labels = full_dataset.targets
        indices = np.arange(len(full_dataset))  # or just list(range(len(full_dataset)))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_ratio,
            random_state=self.seed,
            stratify=labels
        )
        # Create subsets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset   = torch.utils.data.Subset(full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)