"""
Dataset implementations for InfoMAE
Supports: ImageNet-100, CIFAR-100, STL-10
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List


class ImageNet100(Dataset):
    """
    ImageNet-100: subset of ImageNet with 100 classes
    """
    def __init__(self, root: str, split: str = 'train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Use standard ImageNet structure
        split_dir = 'train' if split == 'train' else 'val'
        self.dataset = datasets.ImageFolder(
            os.path.join(root, split_dir),
            transform=None  # We'll apply transform ourselves
        )
        
        # Get 100 random classes (or load from file if exists)
        class_file = os.path.join(root, 'imagenet100_classes.txt')
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                self.selected_classes = [int(x.strip()) for x in f.readlines()]
        else:
            # Randomly select 100 classes
            np.random.seed(42)
            all_classes = list(range(len(self.dataset.classes)))
            self.selected_classes = sorted(np.random.choice(all_classes, 100, replace=False).tolist())
            # Save for reproducibility
            with open(class_file, 'w') as f:
                for cls in self.selected_classes:
                    f.write(f"{cls}\n")
        
        # Filter dataset
        self.indices = [i for i, (_, label) in enumerate(self.dataset.samples) 
                       if label in self.selected_classes]
        
        # Remap labels to 0-99
        self.label_map = {old: new for new, old in enumerate(self.selected_classes)}
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        label = self.label_map[label]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def build_transform(split: str, img_size: int = 224, 
                    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                    augment: bool = True) -> transforms.Compose:
    """Build data transforms"""
    
    if split == 'train' and augment:
        # Training augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation/test transform
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14), interpolation=3),  # 256 for 224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform


def build_dataset(dataset_name: str, root: str, split: str = 'train', 
                 img_size: int = 224, augment: bool = True) -> Dataset:
    """
    Build dataset
    
    Args:
        dataset_name: 'imagenet100', 'cifar100', 'stl10'
        root: data root directory
        split: 'train' or 'val'
        img_size: image size
        augment: whether to use data augmentation
    """
    
    # Build transform
    transform = build_transform(split, img_size, augment=augment)
    
    if dataset_name.lower() == 'imagenet100':
        dataset = ImageNet100(root, split, transform)
    
    elif dataset_name.lower() == 'cifar100':
        is_train = (split == 'train')
        dataset = datasets.CIFAR100(
            root=root,
            train=is_train,
            transform=transform,
            download=True
        )
    
    elif dataset_name.lower() == 'stl10':
        # STL-10 has train/test splits
        split_map = {'train': 'train', 'val': 'test'}
        dataset = datasets.STL10(
            root=root,
            split=split_map[split],
            transform=transform,
            download=True
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


class IndexedDataset(Dataset):
    """
    Wrapper that returns dataset index along with data
    This is needed for epoch-level surprisal caching
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Return: (idx, image, label) or (idx, *data)
        if isinstance(data, tuple):
            return (idx,) + data
        else:
            return idx, data


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 8,
                    shuffle: bool = True, drop_last: bool = True,
                    pin_memory: bool = True, return_index: bool = True) -> DataLoader:
    """
    Build dataloader
    
    Args:
        dataset: PyTorch dataset
        batch_size: batch size
        num_workers: number of workers
        shuffle: whether to shuffle
        drop_last: whether to drop last incomplete batch
        pin_memory: whether to pin memory
        return_index: whether to return dataset index (for epoch caching)
    """
    
    # Wrap dataset to return index if needed
    if return_index:
        dataset = IndexedDataset(dataset)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return loader


class SaliencyDataset(Dataset):
    """
    Dataset for saliency evaluation (MIT300, OSIE)
    """
    def __init__(self, root: str, dataset_name: str = 'mit300', transform=None):
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transform
        
        # Load image paths
        img_dir = os.path.join(root, dataset_name, 'images')
        saliency_dir = os.path.join(root, dataset_name, 'maps')
        
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.saliency_maps = sorted([os.path.join(saliency_dir, f) for f in os.listdir(saliency_dir)
                                     if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        assert len(self.images) == len(self.saliency_maps), \
            f"Mismatch: {len(self.images)} images vs {len(self.saliency_maps)} saliency maps"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        saliency = Image.open(self.saliency_maps[idx]).convert('L')  # Grayscale
        
        if self.transform:
            img = self.transform(img)
            saliency = transforms.ToTensor()(saliency)
        
        return img, saliency, self.images[idx]


def collate_fn_saliency(batch):
    """Custom collate function for variable-sized saliency images"""
    images, saliency_maps, paths = zip(*batch)
    return list(images), list(saliency_maps), list(paths)

